from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import os
import glob
import copy
from tqdm import tqdm
import pickle


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    train_loss = []
    val_loss = []
    random_indices_train = np.random.choice(len(dataset['output_train_features']),
                                      len(dataset['output_train_features']), replace=False)
    random_indices_val = np.random.choice(len(dataset['output_val_features']),
                                            len(dataset['output_val_features']), replace=False)
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Iterate over data.
        for phase in ['output_train_features', 'output_val_features']:
            scheduler.step()
            model.train()  # Set model to training mode
            running_loss = 0.0
            if phase == 'output_train_features':
                random_indices = random_indices_train
            else:
                random_indices = random_indices_val

            for i in tqdm(random_indices):
                inputs = dataset[phase][i]['feature']
                labels = dataset[phase][i]['annotation']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                    outputs = model.classifier(inputs)
                    labels = torch.Tensor(labels).unsqueeze(0)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset[phase])
            if phase == 'output_train_features':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:4f}'.format(best_loss))
    #visualize(train_loss)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(model, criterion):
    model.eval()
    prediction_rescale_annotations = []
    crop_sample = CropFunctionTest(test_dataset[0])
    out_1024 = FeatureExtractor(crop_sample[0]['rescale_img'], crop_sample[1]['rescale_img'])
    test_data = []
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            if i == 0:
                inputs =  out_1024
                outputs = model.classifier(inputs)
                prediction_rescale_annotations.append(outputs.tolist()[0])
            else:
                predicted_crop_ann = ReverseRescaleAnnotation(crop_sample[1]['crop_img'],
                                                              prediction_rescale_annotations[i-1])
                predicted_ann = ReverseCropAnnotation(predicted_crop_ann, crop_sample[1]['crop_img'].shape,
                                                      test_dataset[i][0]['annotation'])
                loss_list.append(criterion(torch.Tensor(predicted_ann).unsqueeze(0),
                                           torch.Tensor([float(x) for x in test_dataset[i][0]['annotation']]).unsqueeze(0)))
                test_data.append(({'image': test_dataset[i][0]['image'], 'annotation': predicted_ann},
                                  {'image': test_dataset[i][1]['image'], 'annotation': test_dataset[i][1]['annotation']}))
                crop_sample = CropFunctionTest(test_data[i-1])
                inputs = FeatureExtractor(crop_sample[0]['rescale_img'], crop_sample[1]['rescale_img'])
                outputs = model.classifier(inputs)
                prediction_rescale_annotations.append(outputs.tolist()[0])
    return test_data


def CropFunctionTest(tp):
    image1 = tp[0]['image']
    annotation1 = tp[0]['annotation']
    image2 = tp[1]['image']
    annotation2 = tp[1]['annotation']
    sample1 = {'image': image1, 'annotation': annotation1}
    sample2 = {'image': image2, 'annotation': annotation2}
    x11, y11, x21, y21 = [float(i) for i in annotation1]
    enlarged_x11 = (x11 - ((x21 - x11) / 2))
    enlarged_y11 = (y11 - ((y21 - y11) / 2))
    enlarged_x21 = (x21 + ((x21 - x11) / 2))
    enlarged_y21 = (y21 + ((y21 - y11) / 2))
    output_size = ((enlarged_y21 - enlarged_y11), (enlarged_x21 - enlarged_x11))
    crop_transform = Crop(output_size)
    cropped_image1 = crop_transform(sample1)
    cropped_image2 = crop_transform(sample2)
    scale = Rescale((224, 224))
    resized_image1 = scale(cropped_image1)
    resized_image2 = scale(cropped_image2)
    cropped_image1.update(resized_image1)
    cropped_image2.update(resized_image2)
    sample = ({'crop_img': cropped_image1['crop_img'], 'rescale_img': cropped_image1['rescale_img']},
              {'crop_img': cropped_image2['crop_img'], 'rescale_img': cropped_image2['rescale_img']})
    return sample


def FeatureExtractor(t0, t1):
    output_t0 = model_ft.features(normalizer(t0).unsqueeze(0))
    output_t1 = model_ft.features(normalizer(t1).unsqueeze(0))
    avg2d = nn.AvgPool2d(5, stride=4)
    output_t0 = avg2d(output_t0).view(1, 512)
    output_t1 = avg2d(output_t1).view(1, 512)
    output_1024 = torch.cat((output_t0, output_t1), 1)
    return output_1024


def read_annotation_file(phase):
    path = sorted(glob.glob('dataset/annotations/*'))
    train_path = sorted(glob.glob('dataset/videos/train/*'))
    val_path = sorted(glob.glob('dataset/videos/val/*'))
    test_path = sorted(glob.glob('dataset/videos/test/*'))
    t_path = [x.split('train/')[1] for x in train_path]
    v_path = [x.split('val/')[1] for x in val_path]
    ts_path = [x.split('test/')[1] for x in test_path]
    ann = []

    if phase == 'train':
        for i in path:
            if i.split('annotations/')[1].split('.')[0] in t_path:
                f = open(i, 'r')
                contents = f.read()
                annotations = []
                video_name = i.split('/')[-1].split('.')[0]
                for j in contents.split('\n'):
                    if j != '':
                        temp = video_name + '/' + (('0' * (8 - len(j.split(' ')[0]))) + j.split(' ')[0] + '.jpg')
                        annotations.append([temp] + j.split(' ')[1:])
                ann += annotations
    elif phase == 'val':
        for i in path:
            if i.split('annotations/')[1].split('.')[0] in v_path:
                f = open(i, 'r')
                contents = f.read()
                annotations = []
                video_name = i.split('/')[-1].split('.')[0]
                for j in contents.split('\n'):
                    if j != '':
                        temp = video_name + '/' + (('0' * (8 - len(j.split(' ')[0]))) + j.split(' ')[0] + '.jpg')
                        annotations.append([temp] + j.split(' ')[1:])
                ann += annotations
    elif phase == 'test':
        for i in path:
            if i.split('annotations/')[1].split('.')[0] in ts_path:
                f = open(i, 'r')
                contents = f.read()
                annotations = []
                video_name = i.split('/')[-1].split('.')[0]
                for j in contents.split('\n'):
                    if j != '':
                        temp = video_name + '/' + (('0' * (8 - len(j.split(' ')[0]))) + j.split(' ')[0] + '.jpg')
                        annotations.append([temp] + j.split(' ')[1:])
                ann += annotations
    return ann


class CustomDataset(Dataset):
    def __init__(self, ann_file, root_dir, transform=None):
        self.ann = ann_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.root_dir, self.ann[idx][0])
        image1 = Image.open(img_name1)
        annotation1 = self.ann[idx][1:]
        img_name2 = os.path.join(self.root_dir, self.ann[idx+1][0])
        image2 = Image.open(img_name2)
        annotation2 = self.ann[idx+1][1:]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        sample1 = {'image': image1, 'annotation': annotation1}
        sample2 = {'image': image2, 'annotation': annotation2}
        sample = (sample1, sample2)
        return sample


class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']
        new_h, new_w = self.output_size
        new_h = int(new_h)
        new_w = int(new_w)
        top = int(float(annotation[1]))
        left = int(float(annotation[0]))
        image = image[ : , top: top + new_h,
                      left: left + new_w]
        ann_crop = 2 * [left, top]
        annotation = [float(a) - float(b) for a, b in zip(annotation, ann_crop)]

        return {'crop_img': image, 'crop_ann': annotation}


def ReverseCropAnnotation(crop_ann, crop_image_size, original_annotation):
    output_size = crop_image_size
    new_h, new_w = output_size[1:]
    new_h = int(new_h)
    new_w = int(new_w)
    top = int(float(original_annotation[1]))
    left = int(float(original_annotation[0]))
    ann_crop = 2 * [left, top]
    annotation = [float(a) + float(b) for a, b in zip(crop_ann, ann_crop)]
    return annotation


def ReverseRescaleAnnotation(crop_image, rescale_ann):
    w = crop_image.shape[1]
    h = crop_image.shape[2]
    output_size = (224, 224)
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)
    ann_scale = 2 * [new_w / w, new_h / h]
    annotation = [float(a) / float(b) for a, b in zip(rescale_ann, ann_scale)]
    return annotation


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        toPIL = transforms.ToPILImage()
        image, annotation = toPIL(sample['crop_img']), sample['crop_ann']

        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        transform_resize = transforms.Resize((new_h, new_w), interpolation=Image.NEAREST)
        toTensor_transform = transforms.ToTensor()
        img = toTensor_transform(transform_resize(image))
        ann_scale = 2 * [new_w / w, new_h / h]
        annotation = [float(a)*float(b) for a, b in zip(annotation, ann_scale)]

        return {'rescale_img': img, 'rescale_ann': annotation}


class CropDataset(Dataset):
    def __init__(self, ann_file, root_dir, transform=None):
        self.ann = ann_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.root_dir, self.ann[idx][0])
        image1 = Image.open(img_name1)
        annotation1 = self.ann[idx][1:]
        img_name2 = os.path.join(self.root_dir, self.ann[idx + 1][0])
        image2 = Image.open(img_name2)
        annotation2 = self.ann[idx + 1][1:]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        sample1 = {'image': image1, 'annotation': annotation1}
        sample2 = {'image': image2, 'annotation': annotation2}
        x11, y11, x21, y21 = [float(i) for i in annotation1]
        enlarged_x11 = (x11 - ((x21 - x11) / 2))
        enlarged_y11 = (y11 - ((y21 - y11) / 2))
        enlarged_x21 = (x21 + ((x21 - x11) / 2))
        enlarged_y21 = (y21 + ((y21 - y11) / 2))
        output_size = ((enlarged_y21 - enlarged_y11), (enlarged_x21 - enlarged_x11))
        crop_transform = Crop(output_size)
        cropped_image1 = crop_transform(sample1)
        cropped_image2 = crop_transform(sample2)
        scale = Rescale((224, 224))
        resized_image1 = scale(cropped_image1)
        resized_image2 = scale(cropped_image2)
        cropped_image1.update(resized_image1)
        cropped_image2.update(resized_image2)
        sample = (cropped_image1, cropped_image2)
        return sample


anno_train_file = read_annotation_file('train')
anno_val_file = read_annotation_file('val')
anno_test_file = read_annotation_file('test')
train_dataset = CustomDataset(ann_file=anno_train_file, root_dir='dataset/videos/train',
                              transform=transforms.ToTensor())
val_dataset = CustomDataset(ann_file=anno_val_file, root_dir='dataset/videos/val',
                              transform=transforms.ToTensor())
cropped_train_dataset = CropDataset(ann_file=anno_train_file, root_dir='dataset/videos/train',
                                    transform=transforms.ToTensor())
cropped_val_dataset = CropDataset(ann_file=anno_val_file, root_dir='dataset/videos/val',
                                    transform=transforms.ToTensor())
test_dataset = CustomDataset(ann_file=anno_test_file, root_dir='dataset/videos/test',
                              transform=transforms.ToTensor())


""" Model Preparation """

filename = 'vgg_16_model.pickle'
exist = os.path.isfile(filename)
if exist:
    print(filename, ' found!!')
    print()
    file = open(filename, 'rb')
    model_ft = pickle.load(file)
    file.close()
else:
    print(filename, ' not found!!')
    print()
    model_ft = models.vgg16(pretrained=True)
    file = open(filename, 'wb')
    pickle.dump(model_ft, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

model_ft.eval()

for p in model_ft.parameters():
    p.requires_grad = False
normalizer = transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])




""" Features Extraction and Saving """

filename = 'output_1024_features.pickle'
exist = os.path.isfile(filename)
if exist:
    print(filename, ' found!!')
    print()
    file = open(filename, 'rb')
    dataset = pickle.load(file)
    file.close()
else:
    print(filename, ' not found!!')
    print()
    dataset = {'output_train_features' : [], 'output_val_features' : []}
    for i, (t0, t1) in tqdm(enumerate(cropped_train_dataset)):
        output_t0 = model_ft.features(normalizer(t0['rescale_img']).unsqueeze(0))
        output_t1 = model_ft.features(normalizer(t1['rescale_img']).unsqueeze(0))
        avg2d = nn.AvgPool2d(5, stride=4)
        output_t0 = avg2d(output_t0).view(1, 512)
        output_t1 = avg2d(output_t1).view(1, 512)
        output_1024 = {'feature' : torch.cat((output_t0, output_t1), 1), 'annotation' : t0['rescale_ann']}
        dataset['output_train_features'].append(output_1024)
    for i, (t0, t1) in tqdm(enumerate(cropped_val_dataset)):
        output_t0 = model_ft.features(normalizer(t0['rescale_img']).unsqueeze(0))
        output_t1 = model_ft.features(normalizer(t1['rescale_img']).unsqueeze(0))
        avg2d = nn.AvgPool2d(5, stride=4)
        output_t0 = avg2d(output_t0).view(1, 512)
        output_t1 = avg2d(output_t1).view(1, 512)
        output_1024 = {'feature' : torch.cat((output_t0, output_t1), 1), 'annotation' : t0['rescale_ann']}
        dataset['output_val_features'].append(output_1024)
    file = open(filename, 'wb')
    pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()



""" Change Model's Classifier Settings """
fc_list = list(model_ft.classifier.children())
fc_list.clear()
fc_list.append(torch.nn.Linear(1024, 1024))
fc_list.append(torch.nn.ReLU())
fc_list.append(torch.nn.Linear(1024, 1024))
fc_list.append(torch.nn.ReLU())
fc_list.append(torch.nn.Linear(1024, 4))
new_classifier = torch.nn.Sequential(*fc_list)
model_ft.classifier = new_classifier

criterion = nn.MSELoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_list = []

""" Model Loading from pickle file """
filename = 'train_model_lr01_e5.pickle'
exist = os.path.isfile(filename)
if exist:
    print(filename, ' found!!')
    print()
    file = open(filename, 'rb')
    model_ft = pickle.load(file)
    file.close()
else:
    print(filename, ' not found!!')
    print()
    model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=5)
    file = open(filename, 'wb')
    pickle.dump(model_ft, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


""" test Loading from pickle file """
filename = 'test_dataset_lr01_e5.pickle'
loss_filename = 'test_dataset_lr01_e5_loss.pickle'
exist = os.path.isfile(filename)
if exist:
    print(filename, ' found!!')
    print()
    file = open(filename, 'rb')
    loss_file = open(loss_filename, 'rb')
    test_data = pickle.load(file)
    loss_list = pickle.load(loss_file)
    file.close()
else:
    print(filename, ' not found!!')
    print()
    loss_list = []
    test_data = test(model_ft, criterion)
    file = open(filename, 'wb')
    loss_file = open(loss_filename, 'wb')
    pickle.dump(test_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(loss_list, loss_file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

