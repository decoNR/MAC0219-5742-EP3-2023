void modify_hue(png_bytep h_image,
                int width,
                int height,
                size_t image_size,
                double hue_diff) {
    double c = cos(2 * M_PI * hue_diff);
    double s = sin(2 * M_PI * hue_diff);
    double one_third = 1.0 / 3.0;
    double sqrt_third = sqrt(one_third);

    // Matriz A compoe as operacoes de
    // conversao de RGB para HSV, mudanca de hue,
    // e conversao de HSV de volta para RGB
    // (new_r, new_g, new_b)' = A * (r, g, b)'
    // https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color

    double a11 = c + one_third * (1.0 - c);
    double a12 = one_third * (1.0 - c) - sqrt_third * s;
    double a13 = one_third * (1.0 - c) + sqrt_third * s;
    double a21 = a13; double a22 = a11; double a23 = a12;
    double a31 = a12; double a32 = a13; double a33 = a11;

    double A[9] = {a11, a12, a13, a21, a22, a23, a31, a32, a33};
    double *d_A;

    // Alocação de memória para matriz A na GPU
    checkErrors(cudaMalloc((void **)&d_A, sizeof(double) * 9), "Alocacao da matriz A no device");

    // Copia da matriz A para a GPU
    checkErrors(cudaMemcpy(d_A, A, sizeof(double) * 9, cudaMemcpyHostToDevice), "Copia da matriz A para o device");

    png_bytep d_image;
    size_t d_image_size = image_size;

    // Alocação de memória para a imagem na GPU
    checkErrors(cudaMalloc((void **)&d_image, d_image_size), "Alocacao da imagem no device");

    // Copia da imagem para a GPU
    checkErrors(cudaMemcpy(d_image, h_image, d_image_size, cudaMemcpyHostToDevice), "Copia da imagem para o device");

    // Determinar as dimensões do grid e dos blocos
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);

    // Chamar o kernel CUDA para processar a imagem em paralelo
    modify_hue_kernel<<<dim_grid, dim_block>>>(d_image, width, height, d_A);
    checkErrors(cudaGetLastError(), "Lancamento do kernel");

    // Copia da imagem de volta para o host
    checkErrors(cudaMemcpy(h_image, d_image, d_image_size, cudaMemcpyDeviceToHost), "Copia da imagem para o host");

    // Liberar memória da GPU
    cudaFree(d_A);
    cudaFree(d_image);
}
