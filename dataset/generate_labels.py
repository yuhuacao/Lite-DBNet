import os
import json


def gen_det_label(root_path, input_dir, out_label):
    with open(out_label, 'w') as out_file:
        for label_file in os.listdir(input_dir):
            img_path = root_path + label_file[3:-4] + ".jpg"
            label = []
            with open(os.path.join(input_dir, label_file), 'r') as f:
                for line in f.readlines():
                    tmp = line.strip().replace('\ufeff', '').split(',')
                    points = tmp[:8]
                    s = []
                    for i in range(0, len(points), 2):
                        b = points[i:i + 2]
                        b = [int(t) for t in b]
                        s.append(b)
                    result = {"transcription": tmp[8], "points": s}
                    label.append(result)

            out_file.write(img_path + '\t' + json.dumps(
                label, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    image_path = 'test/image/'
    label_path = 'test/gt/'
    output_label = 'test_labels.txt'
    gen_det_label(image_path, label_path, output_label)
