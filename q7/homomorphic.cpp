/**
 * Questão 7
 * Nome: Manoel Benedito Neto 
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define RADIUS 20

/* Variáveis base */
cv::Mat image_input, image_output, tmp;
cv::Mat padding, high_pass;
cv::Mat_<float> image_real, mat_zero, mat_ones;
cv::Mat image_complex;
std::vector<cv::Mat> fill;

int gammaL_slider = 2, gammaH_slider = 20, C_slider = 1, D0_slider = 5;
const int gammaL_max = 10, gammaH_max = 50, C_max = 100, D0_max = 200;
double gammaL, gammaH, C, D0;
int dft_M, dft_N;

/**
 * Essa função é responsável por modificar os valores dos quadrantes da imagem de entrada.
*/
void dft_change(cv::Mat& image) {
    // Quadrantes e matriz temporária
    cv::Mat tmp, A, B, C, D;

    // Ajusta tamanho da imagem/matriz
    // Desse modo, ao percorrer a matriz e realizar as modificações nos quadrantes, não se corre
    // o risco de haver partes faltantes
    image = image(cv::Rect(0, 0, image.cols & -2, image.rows & -2));
    int cx = image.cols / 2;
    int cy = image.rows / 2;
    
    // Por fim, reestrutura-se os quadrantes da transformada (A e D - B e C)
    A = image(cv::Rect(0, 0, cx, cy));
    D = image(cv::Rect(cx, cy, cx, cy));
    A.copyTo(tmp);
    D.copyTo(A);
    tmp.copyTo(D);

    B = image(cv::Rect(cx, 0, cx, cy));
    C = image(cv::Rect(0, cy, cx, cy));
    C.copyTo(tmp);
    B.copyTo(C);
    tmp.copyTo(B);
}


/**
 * Responsável por de fato aplica a filtragem sob a imagem de acordo com a aplicação do filtro passa alta
 * por meio da aplicação da função de transferência sob a imagem.
*/
void filter() {
    // Inicializa-se a matriz complexa para zeros e uns
    mat_zero = cv::Mat_<float>::zeros(padding.size());
    mat_ones = cv::Mat_<float>::zeros(padding.size());
    image_complex = cv::Mat(padding.size(), CV_32FC2, cv::Scalar(0));

    // Array que irá compor a matriz desejada
    fill.clear();
    // cria a compoente real
    image_real = cv::Mat_<float>(padding);


    //log
    image_real += cv::Scalar::all(1);
    cv::log(image_real,image_real);
        
    // insere as duas componentes no array de matrizes
    fill.push_back(image_real);
    fill.push_back(mat_zero);

    // combina o array de matrizes em uma unica
    // componente complexa
    cv::merge(fill, image_complex);

    // calcula o dft
    cv::dft(image_complex, image_complex);
    // realiza a troca de quadrantes
    dft_change(image_complex);
    cv::resize(image_complex,image_complex,padding.size());
    cv::normalize(image_complex,image_complex,0,1,cv::NORM_MINMAX);

    // a função de transferencia (filtro de frequencia) deve ter o
    // mesmo tamanho e tipo da matriz complexa
    high_pass = image_complex.clone();

    // cria uma matriz temporária para criar as componentes real
    // e imaginaria do filtro passa alta gaussiano
    tmp = cv::Mat(dft_M, dft_N, CV_32F);
    float D;
    // prepara o filtro passa-alta ideal
    for (int i = 0; i < dft_M; i++) {
        for (int j = 0; j < dft_N; j++) {
        D = (i-dft_M/2)*(i-dft_M/2) + (j-dft_N/2)*(j-dft_N/2);
        tmp.at<float>(i,j) = (gammaH - gammaL)*(1 - exp(-C*( D / (D0*D0) ))) + gammaL; 
        }
    }

    // cria a matriz com as componentes do filtro e junta
    // ambas em uma matriz multicanal complexa
    cv::Mat comps[] = {tmp, tmp};
    cv::merge(comps, 2, high_pass); 

    // aplica o filtro de frequencia
    cv::mulSpectrums(image_complex, high_pass, image_complex, 0);

    // troca novamente os quadrantes
    dft_change(image_complex);

    // calcula a DFT inversa
    cv::idft(image_complex, image_complex);

    // limpa o array de preenchimento
    fill.clear();

    // separa as partes real e imaginaria da
    // imagem filtrada
    cv::split(image_complex, fill);

    cv::exp(fill[0],fill[0]);
    // normaliza a parte real para exibicao
    cv::normalize(fill[0], fill[0], 0, 1, cv::NORM_MINMAX);
    image_output = fill[0].clone();
}

/**
 * Responsável por capturar e aplicar os valores dispostos em uma trackbar interativa sob a imagem
 * de entrada.
 * Função de callback.
*/
void on_trackbar(int, void*){
    C = (double) C_slider;
    D0 = (double) D0_slider;
    gammaL = (double) gammaL_slider/10;
    gammaH = (double) gammaH_slider/10;
    filter();
    cv::imshow("Homomorphic", image_output);
}


int main(int argc, char** argv) {
    // Lê e mostra em tela a imagem de entrada
    image_input = cv::imread(argv[1],cv::IMREAD_GRAYSCALE); 
    cv::namedWindow("Homomorphic",cv::WINDOW_NORMAL);
    cv::imshow("Original",image_input);
    cv::waitKey();    
    
    // Captura-se os tamanhos ótimos para implementar a FFT
    dft_M = cv::getOptimalDFTSize(image_input.rows);
    dft_N = cv::getOptimalDFTSize(image_input.cols);

    // Realiza o espaçamento de borda
    cv::copyMakeBorder(image_input, padding, 0, dft_M - image_input.rows, 0,
                        dft_N - image_input.cols, cv::BORDER_CONSTANT,
                        cv::Scalar::all(0));

    char TrackbarName[50]; // Nome da trackbar que será implementada para cada um dos valores que serão
                           // alterados na aplicação do filtro

    sprintf( TrackbarName, "Gamma L x %d", gammaL_max );
    cv::createTrackbar( TrackbarName, "Homomorphic", &gammaL_slider, gammaL_max, on_trackbar);

    sprintf( TrackbarName, "Gamma H x %d", gammaH_max );
    cv::createTrackbar( TrackbarName, "Homomorphic", &gammaH_slider, gammaH_max, on_trackbar);

    sprintf( TrackbarName, "C x %d", C_max );
    cv::createTrackbar( TrackbarName, "Homomorphic", &C_slider, C_max, on_trackbar);

    sprintf( TrackbarName, "D0 x %d", D0_max );
    cv::createTrackbar( TrackbarName, "Homomorphic", &D0_slider, D0_max, on_trackbar);

    cv::waitKey(0);
    return 0;
}