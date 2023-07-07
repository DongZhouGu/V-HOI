import json, os, pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

spatial_relation = ['above', 'next_to', 'behind', 'away', 'towards', 'in_front_of', 'inside', 'beneath']


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.uint8):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(annot_file, out_dir, filename):
    with open(os.path.join(out_dir, filename), 'w') as outfile:
        outfile.write(json.dumps(annot_file, cls=MyEncoder))


def generate_corr(annotations):
    coco_ids = list(range(78))
    verb_ids = [
        0, 1, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49]
    matrix = np.zeros((len(verb_ids), len(coco_ids)))

    for annot in annotations:
        bboxes = annot['annotations']
        hois = annot['hoi_annotation']
        for hoi in hois:
            obj_id = hoi['object_id']
            matrix[verb_ids.index(int(hoi['category_id'])), coco_ids.index(int(bboxes[obj_id]['category_id']))] = 1
    np.save('corre_mat_vhoi.npy', matrix)


def convert_vidor_to_ppdm_label(annot_dir):
    human_categories = ['adult', 'child', 'baby']
    with open('pred_categories.json', 'r') as f:
        pred_categories = json.load(f)
    with open('obj_to_idx.pkl', 'rb') as f:
        obj_to_idx = pickle.load(f)
    with open('pred_to_idx.pkl', 'rb') as f:
        pred_to_idx = pickle.load(f)
    frame_annots_hoi = []
    img_indx = []
    used_video_dict = set()
    for folder in tqdm(os.listdir(annot_dir)):
        for video_json in os.listdir(os.path.join(annot_dir, folder)):
            with open(os.path.join(annot_dir, folder, video_json), 'r') as f:
                annot = json.load(f)

            if abs(annot['fps'] - 29.97) < 0.1:
                fps = 30
            elif annot['fps'] - 24 < 1.01:  # fps 24, 25
                fps = 24
            else:
                import pdb;
                pdb.set_trace()

            for rel in annot['relation_instances']:
                if annot['subject/objects'][rel['subject_tid']]['category'] in human_categories \
                        and rel[
                    'predicate'] in pred_categories and rel['predicate'] not in spatial_relation:

                    for idx in range(rel['begin_fid'], rel['end_fid']):
                        if (idx + 1 - (fps // 2)) % fps != 0:  # not middle frame
                            continue

                        frame_annot = annot['trajectories'][idx]

                        person_found = object_found = False
                        for ann in frame_annot:
                            if ann['tid'] == rel['subject_tid']:
                                person_annot, person_found = ann, True
                            elif ann['tid'] == rel['object_tid']:
                                object_annot, object_found = ann, True
                            if person_found and object_found:
                                break

                        file_name = folder + '/' + annot['video_id'] + '/' + annot['video_id'] + '_' + str(
                            f'{idx + 1 :06d}' + '.jpg')
                        if file_name not in img_indx:
                            img_indx.append(file_name)
                            frame_annots_hoi.append({'file_name': file_name, 'video_fps': fps,  # annot['fps'],
                                                     'height': annot['height'],
                                                     'width': annot['width'], 'annotations': [], 'hoi_annotation': []})
                            used_video_dict.add(folder + '/' + annot['video_id'])

                        sub_box = [person_annot['bbox']['xmin'], person_annot['bbox']['ymin'],
                                   person_annot['bbox']['xmax'], person_annot['bbox']['ymax']]
                        obj_box = [object_annot['bbox']['xmin'], object_annot['bbox']['ymin'],
                                   object_annot['bbox']['xmax'], object_annot['bbox']['ymax']]
                        obj_cate = obj_to_idx['person'] if annot['subject/objects'][rel['object_tid']][
                                                               'category'] in human_categories else obj_to_idx[
                            annot['subject/objects'][rel['object_tid']]['category']]
                        sub_annot = {'bbox': sub_box, 'category_id': obj_to_idx['person'],
                                     'tic_id': person_annot['tid']}
                        obj_annot = {'bbox': obj_box, 'category_id': obj_cate, 'tic_id': object_annot['tid']}
                        f_indx = img_indx.index(file_name)
                        if sub_annot not in frame_annots_hoi[f_indx]['annotations']:
                            frame_annots_hoi[f_indx]['annotations'].append(sub_annot)
                        if obj_annot not in frame_annots_hoi[f_indx]['annotations']:
                            frame_annots_hoi[f_indx]['annotations'].append(obj_annot)
                        sub_id = frame_annots_hoi[f_indx]['annotations'].index(sub_annot)
                        obj_id = frame_annots_hoi[f_indx]['annotations'].index(obj_annot)

                        v_id = pred_to_idx[rel['predicate']]
                        hoi = {'subject_id': sub_id, 'object_id': obj_id, 'category_id': v_id}
                        if hoi not in frame_annots_hoi[f_indx]['hoi_annotation']:
                            frame_annots_hoi[f_indx]['hoi_annotation'].append(hoi)

    return frame_annots_hoi, used_video_dict


def convert_vidor_to_csv(video_dict, annot_dir):
    data = []
    i = 0
    for folder in tqdm(os.listdir(annot_dir)):
        for video_json in os.listdir(os.path.join(annot_dir, folder)):
            video_id = folder + '/' + video_json.split('.')[0]
            if video_id not in video_dict:
                continue
            with open(os.path.join(annot_dir, folder, video_json), 'r') as f:
                annot = json.load(f)
            for j in range(annot['frame_count']):
                data.append(
                    (folder + '/' + annot['video_id'], i, j,
                     os.path.join(folder, annot['video_id'], annot['video_id'] + '_' + str(f'{j + 1:06d}') + '.jpg'),
                     '')
                )
            i += 1

    train_frame_list = pd.DataFrame(data, columns=['original_video_id', 'video_id', 'frame_id', 'path', 'labels'])
    return train_frame_list


train_annot_dir = './anno/train'
val_annot_dir = './anno/val'

train_frame_annots, train_used_video_dict = convert_vidor_to_ppdm_label(train_annot_dir)
save_json(train_frame_annots, '.', 'vidhoi_train.json')
val_frame_annots, val_used_video_dict = convert_vidor_to_ppdm_label(val_annot_dir)
save_json(val_frame_annots, '.', 'vidhoi_val.json')

train_frame_list = convert_vidor_to_csv(train_used_video_dict, train_annot_dir)
train_frame_list.to_csv('train.csv', sep=' ', index=False)
val_frame_list = convert_vidor_to_csv(val_used_video_dict, val_annot_dir)
val_frame_list.to_csv('val.csv', sep=' ', index=False)
generate_corr(train_frame_annots)
