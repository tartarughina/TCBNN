import os

def main():
    alcf_path = '/eagle/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train>/'
    with open('imagenet_files.txt') as f:
        lines = f.readlines()

        with open('polaris_imagenet_files.txt', 'w') as o:
            for line in lines:
                path, val = line.split('_', maxsplit=1)

                dir = path.split('/')[-1]
                o.write(alcf_path + dir + '/' + dir + '_' + val.strip() + '\n')

if __name__ == "__main__":
    main()
