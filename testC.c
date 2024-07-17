#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int i, j;

// ����ļ��Ƿ����
int file_exists(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file) {
        fclose(file);
        return 1;
    }
    return 0;
}

// ���ļ��ж�ȡ����������
void read_floats_from_file(const char *filename, float *array, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("�޷����ļ� %s\n", filename);
        exit(1);
    }
    for (i = 0; i < size; i++) {
        fscanf(file, "%f,", &array[i]);
    }
    fclose(file);
}

// ��ȡMNISTͼ������
unsigned char* read_mnist_images(const char* file_path, int* number_of_images, int* image_size) {
    printf("Attempting to open image file: %s\n", file_path);
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        printf("�޷����ļ� %s\n", file_path);
        return NULL;
    }

    // ��ȡ�ļ�ͷ
    int magic_number = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = __builtin_bswap32(magic_number); // ת��Ϊ�����ֽ���

    fread(number_of_images, sizeof(*number_of_images), 1, file);
    *number_of_images = __builtin_bswap32(*number_of_images);

    int rows = 0;
    fread(&rows, sizeof(rows), 1, file);
    rows = __builtin_bswap32(rows);

    int cols = 0;
    fread(&cols, sizeof(cols), 1, file);
    cols = __builtin_bswap32(cols);

    *image_size = rows * cols;

    // ��ȡͼ������
    unsigned char* images = (unsigned char*)malloc((*number_of_images) * (*image_size));
    fread(images, sizeof(unsigned char), (*number_of_images) * (*image_size), file);

    fclose(file);
    return images;
}

// ��ȡMNIST��ǩ����
unsigned char* read_mnist_labels(const char* file_path, int* number_of_labels) {
    printf("Attempting to open label file: %s\n", file_path);
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        printf("�޷����ļ� %s\n", file_path);
        return NULL;
    }

    // ��ȡ�ļ�ͷ
    int magic_number = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = __builtin_bswap32(magic_number); // ת��Ϊ�����ֽ���

    fread(number_of_labels, sizeof(*number_of_labels), 1, file);
    *number_of_labels = __builtin_bswap32(*number_of_labels);

    // ��ȡ��ǩ����
    unsigned char* labels = (unsigned char*)malloc((*number_of_labels) * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), (*number_of_labels), file);

    fclose(file);
    return labels;
}

// ��һ��ͼ������
void normalize_image(unsigned char* image, float* normalized_image, int image_size) {
    for (i = 0; i < image_size; i++) {
        normalized_image[i] = ((float)image[i] / 255.0 - 0.1307) / 0.3081;
    }
}

// ���Բ�ǰ�򴫲�
void linear_forward(float* input, float* output, float* weights, float* bias, int input_size, int output_size) {
    for (i = 0; i < output_size; i++) {
        output[i] = 0;
        for (j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
        if (bias != NULL) {
            output[i] += bias[i];
        }
    }
}

int main() {
    srand(42); // �����������

    const char* image_path = "D:\\MNIST_data\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
    const char* label_path = "D:\\MNIST_data\\train-images-idx3-ubyte\\train-images.idx3-ubyte";

    // ����ļ��Ƿ����
    if (!file_exists(image_path)) {
        printf("Image file does not exist: %s\n", image_path);
        return -1;
    }

    if (!file_exists(label_path)) {
        printf("Label file does not exist: %s\n", label_path);
        return -1;
    }

    int num_images, image_size;
    unsigned char* images = read_mnist_images(image_path, &num_images, &image_size);
    if (images == NULL) {
        return -1;
    }

    int num_labels;
    unsigned char* labels = read_mnist_labels(label_path, &num_labels);
    if (labels == NULL) {
        free(images);
        return -1;
    }

    // ֻ�����һ��ͼ��
    unsigned char* image = images;
    float normalized_image[image_size];
    normalize_image(image, normalized_image, image_size);

    // ��ӡԤ������ͼ������
    printf("Preprocessed Image Data Shape: [%d]\n", image_size);
    printf("Preprocessed Image Data: ");
    for (i = 0; i < image_size; i++) {
        printf("%.6f ", normalized_image[i]);
    }
    printf("\n");

    // �������ز��СΪ128
    int input_size = 784;
    int hidden_size = 128;
    float weights[input_size * hidden_size];
    float bias[hidden_size];

    // ���ļ��ж�ȡȨ�غ�ƫ��
    read_floats_from_file("D:\\MNIST test\\fc1_weights.txt", weights, input_size * hidden_size);
    read_floats_from_file("D:\\MNIST test\\fc1_bias.txt", bias, hidden_size);

    float output[hidden_size];
    linear_forward(normalized_image, output, weights, bias, input_size, hidden_size);

    // ��ӡ���Բ����
    printf("Output After Linear Layer:\n");
    for (i = 0; i < hidden_size; i++) {
        printf("%.8f ", output[i]);
    }
    printf("\n");

    // �ͷ��ڴ�
    free(images);
    free(labels);

    return 0;
}

