# Generate cross-validation folds for CT-ICH

import os
import shutil
from collections import defaultdict

import pandas as pd

IMAGE_DIR = "data/CT-ICH/data/image"
MASK_DIR = "data/CT-ICH/data/label"
CSV_PATH = "data/CT-ICH/data/hemorrhage_diagnosis_raw_ct.csv"
FOLD_DIR = "data/CT-ICH/data/fold"
NUM_FOLD = 5

if __name__ == '__main__':
    # get labels for each patient
    patient_labels = defaultdict(bool)
    num_slices = defaultdict(int)
    for _, row in pd.read_csv(CSV_PATH).iterrows():
        patient_id = row['PatientNumber']
        positive = row['No_Hemorrhage'] == 0
        patient_labels[patient_id] = max(patient_labels[patient_id], positive)
        num_slices[patient_id] += 1

    # separate positive patients from negatives
    positive_patients = []
    negative_patients = []
    for patient_id, label in patient_labels.items():
        if label:
            positive_patients.append(patient_id)
        else:
            negative_patients.append(patient_id)

    def _copy_samples(patient_ids, target_dir):
        image_dir = os.path.join(target_dir, "image")
        label_dir = os.path.join(target_dir, "label")
        if not os.path.exists(target_dir):
            os.makedirs(image_dir)
            os.makedirs(label_dir)
        for _patient_id in patient_ids:
            for slice_id in range(num_slices[_patient_id]):
                image_filename = f"{_patient_id}_{slice_id}.png"
                try:
                    shutil.copy(os.path.join(IMAGE_DIR, image_filename), image_dir)
                    shutil.copy(os.path.join(MASK_DIR, image_filename), label_dir)
                except FileNotFoundError:
                    continue

    for _list in (positive_patients, negative_patients):
        fold_size = len(_list) // NUM_FOLD
        for k in range(NUM_FOLD):
            start = k * fold_size
            end = (start + fold_size) if k != NUM_FOLD - 1 else len(_list)
            test_ids = _list[start:end]
            train_ids = list(set(_list).difference(set(test_ids)))
            _copy_samples(test_ids, os.path.join(f"{FOLD_DIR}-{k + 1}", "test"))
            _copy_samples(train_ids, os.path.join(f"{FOLD_DIR}-{k + 1}", "train"))
