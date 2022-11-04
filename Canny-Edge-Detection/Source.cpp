/**
 * Instituto Politécnico Nacional
 * Escuela Superior de Cómputo
 *
 * Grupo: 5BM1
 * Materia: Vision artificial
 *
 * @author Angel Gabriel Rodriguez Rodriguez
 *
 * @brief Programa que implementar el metodo de Canny para deteccion de bordes
 *
 * @version 1.0
 * @date 2022-11-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#define PI 3.141596

using namespace cv;
using namespace std;

/**
 * @brief Función para calcular el valor f(x, y) de la funcion de la campana de gauss
 *
 * @param x Valor x de la ecuacion
 * @param y Valor y de la ecuacion
 * @param sigma Valor de sigma de la ecuacion
 * @return float El valor f(x, y)
 */
float gaussian(int x, int y, float sigma) {
	return (1 / (2 * PI * pow(sigma, 2))) * exp(-((x * x) + (y * y)) / (2 * sigma * sigma));
}

/**
 * @brief Función para crear un filtro gausiano en una matriz
 *
 * @param size El tamaño del filtro cuadrado
 * @param sigma El valor de sigma en la funcion de gauss
 * @return vector<vector<float>> Matriz con los valores del filtro gaussiano
 */
vector<vector<float>> createGaussianFilter(int size, float sigma) {

	// Creamos la matriz del filtro
	vector<vector<float>> filter(size, vector<float>(size));
	int middle = (size - 1) / 2;
	int i = 0, j = 0;
	double sum = 0;

	// Recorremos el filtro y por cada valor de (x, y) obtenemos el 
	// valor de f(x, y) 
	for (int x = -middle, i = 0; x <= middle; x++, i++) {
		for (int y = -middle, j = 0; y <= middle; y++, j++) {
			filter[i][j] = gaussian(x, y, sigma);
			sum += filter[i][j];
		}
	}

	// Normalizamos el filtro para evitar valores muy pequeños
	// y que nuestra imagen no tienda a negro absoluto
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			filter[i][j] /= sum;
		}
	}

	return filter;
}

/**
 * @brief Función para generar el gradiente de una imagen usando los operadores de Sobel
 *
 * @param image La imagen a la que se le aplicara el gradiente
 * @return vector<Mat> Vector con todas las imagenes generadas (Gx, Gy, |G|, theta)
 */
vector<Mat> createSobel(Mat image) {
	// Definimos nuestros operadores de Sobel ya preestablecidos
	vector<vector<int>> gxKernel = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	vector<vector<int>> gyKernel = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	// Creamos todsa nuestras matrices de openCv para despues escribir en ellas
	Mat gx = Mat::zeros(image.rows, image.cols, CV_8UC1);
	Mat gy = Mat::zeros(image.rows, image.cols, CV_8UC1);
	Mat gm = Mat::zeros(image.rows, image.cols, CV_8UC1);
	Mat gAngle = Mat::zeros(image.rows, image.cols, CV_32FC1);

	// Recorremos la imagen de entrada
	for (int i = 1; i < image.rows - 1; i++) {
		for (int j = 1; j < image.cols - 1; j++) {

			// Recorremos el kernel por cada pixel y hacemos la multiplicacion
			// punto por punto y sumandolo a una variable
			int sumGx = 0, sumGy = 0;
			for (int x = 0; x < gxKernel.size(); x++) {
				for (int y = 0; y < gxKernel.size(); y++) {
					sumGx += gxKernel[x][y] * image.at<uchar>(i + x - 1, j + y - 1);
					sumGy += gyKernel[x][y] * image.at<uchar>(i + x - 1, j + y - 1);
				}
			}

			// Asignamos dicho valor a cada imagen respectivamente
			gx.at<uchar>(i, j) = sumGx;
			gy.at<uchar>(i, j) = sumGy;
			gm.at<uchar>(i, j) = sqrt(pow(sumGx, 2) + pow(sumGy, 2));
			gAngle.at<float>(i, j) = atan2(sumGy, sumGx);
		}
	}

	// Creamos el arreglo de respuesta que contiene todos los elementos generados
	vector<Mat> response = { gx, gy, gm, gAngle };
	return response;
}

/**
 * @brief Función para obtener el valor maximo y minimo de una imagen dada
 *
 * @param image La imagen en la que se buscara el maximo y minimo
 * @return pair<int, int> Una tupla con lo valores del maximo y minimo respectivamente
 */
pair<int, int> getMinMaxValues(Mat image) {
	// Definimos el valor maximo y minimo
	int min = 255, max = 0;
	// Iteramos toda la matriz en busca del maximo y minimo
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) < min) {
				min = image.at<uchar>(i, j);
			}
			else if (image.at<uchar>(i, j) > max) {
				max = image.at<uchar>(i, j);
			}
		}
	}
	return make_pair(max, min);
}

/**
 * @brief Función para realizar una supresion de no maximos a una imagen
 *
 * @param gAngle La matriz de los valores de los angulos del gradiente
 * @param gm La matriz de los valores de la magnitud del gradiente
 * @return Mat Imagen procesada con los bordes mas delgados
 */
Mat nonMaximumSuppression(Mat gAngle, Mat gm) {

	// Creamos nuestra matriz donde se guardara el valor de la imagen con el NMS aplicado
	Mat nms = Mat::zeros(gm.rows, gm.cols, CV_8UC1);

	// Recorremos toda la matriz de entrada
	for (int i = 1; i < gm.rows - 1; i++) {
		for (int j = 1; j < gm.cols; j++) {

			// Obtenemos el angulo de cada pixel, este nos indica en que direccion los
			// pixeles se asemejan mas a este
			float angle = abs(gAngle.at<float>(i, j)) * (180 / PI);
			int pixel1, pixel2;

			// Definimos los intervalos de angulos para obtener los 2 vecinos 
			// que estan a los lados del pixel procesado

			// Los vecinos son los de la derecha e izquierda
			if ((angle >= 0 && angle < 67.5) || (angle > 157.5 && angle <= 180)) {
				pixel1 = gm.at<uchar>(i, j + 1);
				pixel2 = gm.at<uchar>(i, j - 1);
			}
			// Los vecinos son los las esquinas superior derecha e inferior izquierda
			else if (angle >= 22.5 && angle < 67.5) {
				pixel1 = gm.at<uchar>(i - 1, j + 1);
				pixel2 = gm.at<uchar>(i + 1, j - 1);
			}
			// Los vecinos que estan arriba y abajo
			else if (angle >= 67.5 && angle < 112.5) {
				pixel1 = gm.at<uchar>(i - 1, j);
				pixel2 = gm.at<uchar>(i + 1, j);
			}
			// Los vecinos que estan en las esquinas superior izquierda e inferior derecha
			else if (angle >= 112.5 && angle < 157.5) {
				pixel1 = gm.at<uchar>(i - 1, j - 1);
				pixel2 = gm.at<uchar>(i + 1, j + 1);
			}

			// Si se cumple la consdicion escribimos el valor en la matriz NMS, de lo contrario
			// solo se queda el cero de la matriz cuando la creamos
			if (gm.at<uchar>(i, j) >= pixel2 && gm.at<uchar>(i, j) >= pixel1)
				nms.at<uchar>(i, j) = gm.at<uchar>(i, j);

		}
	}
	return nms;

}

/**
 * @brief Función para realizar la histeresis y completar el metodo de canny
 *
 * @param nms Matris NMS de la imagen a procesar
 * @param highThresholdPercent Valor porcentual del humbral alto
 * @param lowThresholdPercent Valor porcentual del humbral bajo
 * @return Mat Imagen procesada con los bordes mas detallados y simples
 */
Mat cannyMethod(Mat nms, float highThresholdPercent = 0.9, float lowThresholdPercent = 0.35) {
	// Cramos la matriz donde se alojara nuestro resultado
	Mat canny = Mat::zeros(nms.rows, nms.cols, CV_8UC1);

	// Obtenemos el maximo y minimo de nuestra matriz NMS
	pair<int, int> minMax = getMinMaxValues(nms);
	int max = minMax.first;
	int min = minMax.second;

	// Definimos nuestros umbrales
	float highThreshold = max * highThresholdPercent;
	float lowThreshold = highThreshold * lowThresholdPercent;

	// Definimos nuestros 3 valores que van a conformar la imagen final
	int irrelevant = 0;
	int weak = lowThreshold;
	int strong = 255;

	// Iteramos sobre toda la matriz NMS pixel por pixel
	for (int i = 1; i < nms.rows - 1; i++) {
		for (int j = 1; j < nms.cols; j++) {
			// Si nuestro valor esta entre los umbrales entonces se le asigna el valor "debil"
			if (nms.at<uchar>(i, j) > lowThreshold && nms.at<uchar>(i, j) < highThreshold) {
				canny.at<uchar>(i, j) = weak;
			}
			// Si nuestro umbral esta por arriba del umbral elevado entonces se le asigna el valor "fuerte"
			else if (nms.at<uchar>(i, j) >= highThreshold) {
				canny.at<uchar>(i, j) = strong;
			}

		}
	}
	return canny;
}

int main() {

	int sizeKernel = 2;
	// Entrada del tamaño del kernel, siempre y cuando sea par y mayor a 1
	while (sizeKernel % 2 == 0 || sizeKernel <= 1) {
		cout << "Kernel size: ";
		cin >> sizeKernel;
	}
	// Definimos la entrada del sigma por parte del usuario, asi como salcular 
	// la cantidad de bordes por aplicar a la imagen
	int middle = (sizeKernel - 1) / 2;
	float sigma = 0;
	cout << "Sigma: ";
	cin >> sigma;

	// Obtenemos el filtro gaussiano que se genera dinamicamente y se almacena
	// en una matriz
	vector<vector<float>> filter = createGaussianFilter(sizeKernel, sigma);

	// Definimos el nombre de la imagen
	char imageName[] = "Lenna.png";
	Mat image;

	// Leemos la imagen
	image = imread(imageName);
	// Si no se puede leer la imagen mostramos mensaje de error y terminamos el programa
	if (!image.data) {
		cout << "Error al cargar la imagen: " << imageName << endl;
		exit(1);
	}

	// Filas y columnas originales
	int rows = image.rows;
	int columns = image.cols;

	// Filas y columnas de la imagen con bordes extras
	int rowsFiltered = rows + sizeKernel - 1;
	int columnsFiltered = columns + sizeKernel - 1;

	// Creamos la imagen en escala de grises usando la función de openCv
	Mat imageGray(rows, columns, CV_8UC1);
	cvtColor(image, imageGray, COLOR_BGR2GRAY);

	// Creamos las imagenes con bordes extras y la ya filtrada respectivamente
	Mat imageBorders = Mat::zeros(Size(rowsFiltered, columnsFiltered), CV_8UC1);
	Mat imageFiltered = Mat::zeros(Size(rows, columns), CV_8UC1);

	// Añadimos los bordes extra a nuestra imagen para que sea mas facil de procesar
	// cuando pase por el filtro gaussiano
	for (int i = middle; i < middle + rows; i++) {
		for (int j = middle; j < middle + columns; j++) {
			imageBorders.at<uchar>(i, j) = imageGray.at<uchar>(i - middle, j - middle);
		}
	}

	// Aplicamos el filtro gaussiano a nuestra imagen con bordes extra, haciendo la multiplicación
	// punto a punto de matrices de N x N
	for (int i = middle; i < rows; i++) {
		for (int j = middle; j < columns; j++) {
			// Variable para la suma de los valores de la multiplicación punto a punto
			double sum = 0;
			for (int x = 0; x < sizeKernel; x++)
				for (int y = 0; y < sizeKernel; y++)
					sum += imageBorders.at<uchar>(i - middle + x, j - middle + y) * filter[x][y];
			// Al pixel central sobre el que estamos procesando le asignamos la suma de todas las 
			// multiplicaciones punto por punto del kernel con la imagen en tramos
			imageFiltered.at<uchar>(i - middle, j - middle) = (int)sum;
		}
	}

	// Aplicamos los operadores de Sobel, nos devuelve 4 matrices:
	// 0 -> La matriz Gx de la componente x
	// 1 -> La matriz Gy de la componente y
	// 2 -> La matriz |G| de la magnitud
	// 3 -> La matriz theta del angulo
	vector<Mat> sobel = createSobel(imageFiltered);
	Mat gx = sobel[0];
	Mat gy = sobel[1];
	Mat gm = sobel[2];
	Mat gAngle = sobel[3];

	// Aplicamos la supresion no maxima para adelgazar los bordes
	Mat nms = nonMaximumSuppression(gAngle, gm);
	// Finalmente aplicamos canny para obtener la imagen final
	Mat canny = cannyMethod(nms);


	// Imprimimos el filtro en consola
	cout << endl << "Filtro gaussiano" << endl;
	for (int i = 0; i < filter.size(); i++) {
		for (int j = 0; j < filter.size(); j++) {
			cout << filter[i][j] << " \t";
		}
		cout << endl;
	}
	cout << endl;

	// Mostramos todas las imagenes en pantalla
	namedWindow("Imagen original", WINDOW_AUTOSIZE);
	imshow("Imagen original", image);

	namedWindow("Imagen escala de grises", WINDOW_AUTOSIZE);
	imshow("Imagen escala de grises", imageGray);

	namedWindow("Imagen escala de grises con padding", WINDOW_AUTOSIZE);
	imshow("Imagen escala de grises con padding", imageBorders);

	namedWindow("Imagen suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen suavizada", imageFiltered);

	namedWindow("Imagen sobel magnitud", WINDOW_AUTOSIZE);
	imshow("Imagen sobel magnitud", gm);

	namedWindow("Imagen sobel angulo", WINDOW_AUTOSIZE);
	imshow("Imagen sobel angulo", gAngle);

	namedWindow("Imagen sobel gx", WINDOW_AUTOSIZE);
	imshow("Imagen sobel gx", gx);

	namedWindow("Imagen sobel gy", WINDOW_AUTOSIZE);
	imshow("Imagen sobel gy", gy);

	namedWindow("Imagen nms", WINDOW_AUTOSIZE);
	imshow("Imagen nms", nms);

	namedWindow("Imagen Canny", WINDOW_AUTOSIZE);
	imshow("Imagen Canny", canny);


	// Imprimimos el tamaño de todas las imagenes que generamos 
	cout << "Imagen original (rows): " << image.rows << endl;
	cout << "Imagen original (columns): " << image.cols << endl << endl;

	cout << "Imagen escala de grises (rows): " << imageGray.rows << endl;
	cout << "Imagen escala de grises (columns): " << imageGray.cols << endl << endl;

	cout << "Imagen escala de grises con padding (rows): " << imageBorders.rows << endl;
	cout << "Imagen escala de grises con padding (columns): " << imageBorders.cols << endl << endl;

	cout << "Imagen suavizada (rows): " << imageFiltered.rows << endl;
	cout << "Imagen suavizada (columns): " << imageFiltered.cols << endl << endl;

	cout << "Imagen sobel magnitud (rows): " << gm.rows << endl;
	cout << "Imagen sobel magnitud (columns): " << gm.cols << endl << endl;

	cout << "Imagen sobel angulo (rows): " << gy.rows << endl;
	cout << "Imagen sobel angulo (columns): " << gy.cols << endl << endl;

	cout << "Imagen sobel gx (rows): " << gAngle.rows << endl;
	cout << "Imagen sobel gx (columns): " << gAngle.cols << endl << endl;

	cout << "Imagen sobel gy (rows): " << gx.rows << endl;
	cout << "Imagen sobel gy (columns): " << gx.cols << endl << endl;

	cout << "Imagen nms (rows): " << nms.rows << endl;
	cout << "Imagen nms (columns): " << nms.cols << endl << endl;

	cout << "Imagen Canny (rows): " << canny.rows << endl;
	cout << "Imagen Canny (columns): " << canny.cols << endl << endl;

	waitKey(0);
	return 1;
}