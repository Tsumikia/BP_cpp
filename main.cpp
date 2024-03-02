#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <random>


#define INNODE 2
#define HIDENODE 4
#define OUTNODE 1

double rate = 0.8; // ����
double threshold = 1e-4; // �����������
size_t mosttimes = 1e7; // ����������


struct Sample { // ����
	std::vector<double> in, out;

};

struct Node { // ������Ԫģ��
	double value{}, bias{}, bias_delta{};
	std::vector<double> weight, weight_delta; // Ȩֵ
};

namespace utils {

	// �����
	inline double sigmoid(double x) {
		double res = 1.0 / (1.0 + std::exp(-x));
		return res;
	}

	std::vector<double> getFileData(std::string filename) {
		std::vector<double> res;

		std::ifstream in(filename); // �ļ�������
		if (in.is_open()) {
			while (!in.eof()) {
				double buffer;
				in >> buffer;
				res.push_back(buffer);
			}
			in.close();
		}
		else {
			std::cout << "Error in reading " << filename << std::endl;
		}

		return res;
	}

	std::vector<Sample> getTrainData(std::string filename) {
		std::vector<Sample> res;

		std::vector<double> buffer = getFileData(filename);

		for (size_t i = 0; i < buffer.size(); i += INNODE + OUTNODE) {
			Sample tmp;
			for (size_t t = 0; t < INNODE; t++) {
				tmp.in.push_back(buffer[i + t]);
			}
			for (size_t t = 0; t < OUTNODE; t++) {
				tmp.out.push_back(buffer[i + INNODE + t]);
			}
			res.push_back(tmp);
		}

		return res;
	}

	std::vector<Sample> getTestData(std::string filename) {
		std::vector<Sample> res;

		std::vector<double> buffer = getFileData(filename);

		for (size_t i = 0; i < buffer.size(); i += INNODE) {
			Sample tmp;
			for (size_t t = 0; t < INNODE; t++) {
				tmp.in.push_back(buffer[i + t]);
			}
			res.push_back(tmp);
		}
		return res;
	}
}

// ����㣬 ���ز㣬 �����
Node* inputLayer[INNODE], * hideLayer[HIDENODE], * outLayer[OUTNODE];

inline void init() {
	std::mt19937 rd;
	rd.seed(std::random_device()()); // ������Ӳ�����operator()�Ի�ȡ�����

	std::uniform_real_distribution<double> distribution(-1, 1);  // -1 �� 1.0 �����ʵ��

	for (size_t i = 0; i < INNODE; i++) {
		::inputLayer[i] = new Node(); // ��ʼ��INNODE���ڵ�
		for (size_t j = 0; j < HIDENODE; j++) {
			::inputLayer[i]->weight.push_back(distribution(rd)); // ��Ӧ���ǵ�i���ڵ㵽��j�����ز��weight
			::inputLayer[i]->weight_delta.push_back(0.f); // ����ֵһ��ʼ����0
		}
	}

	// ����Ϊ�����ز�ĳ�ʼ��
	for (size_t i = 0; i < HIDENODE; i++) {
		::hideLayer[i] = new Node();
		::hideLayer[i]->bias = distribution(rd);
		for (size_t j = 0; j < OUTNODE; j++) {
			::hideLayer[i]->weight.push_back(distribution(rd));
			::hideLayer[i]->weight_delta.push_back(0.f);
		}
	}

	// �����
	for (size_t i = 0; i < OUTNODE; i++) {
		::outLayer[i] = new Node();
		::outLayer[i]->bias = distribution(rd);
	}
}


/*
	������������ÿ���ڵ��"���Զ�̬�����ĵ���ֵ"
*/
inline void reset_delta() {
	for (size_t i = 0; i < INNODE; i++) {
		::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(), 0.f);
	}

	for (size_t i = 0; i < HIDENODE; i++) {
		::hideLayer[i]->bias_delta = 0.f;
		::hideLayer[i]->weight_delta.assign(::hideLayer[i]->weight_delta.size(), 0.f);
	}

	for (size_t i = 0; i < OUTNODE; i++) {
		::outLayer[i]->bias_delta = 0.f;
	}
}

int main(int argc, char* argv[]) {

	init();

	size_t t = 0;

	std::vector<Sample> train_data = utils::getTrainData("traindata.txt");

	for (size_t times = 0; times < mosttimes; times++) {
		t++;
		// ÿ�ε���ǰ������deltaֵ
		reset_delta();

		// ������
		double error_max = 0.f;


		// ����ÿһ��ѵ������
		for (auto& idx : train_data) {
			// ������ʼ��
			for (size_t i = 0; i < INNODE; i++) {
				::inputLayer[i]->value = idx.in[i];
			}

			// *** ���򴫲� ***
			for (size_t j = 0; j < HIDENODE; j++) {
				double sum = 0; // �����ʽ�ĺ�
				for (size_t i = 0; i < INNODE; i++) {
					// ��i���ڵ��ֵ ���� ��i���ڵ㵽��j���ڵ��Ȩֵ
					sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
				}
				sum -= ::hideLayer[j]->bias; // ��Ҫ������j�����ز�ڵ��ƫ��ֵ

				// Ȼ���j�����ز��ֵ���Ѿ�ȷ��
				// ������ʽ��ͽ�����뼤���
				::hideLayer[j]->value = utils::sigmoid(sum);
			}

			// ����������value
			for (size_t j = 0; j < OUTNODE; j++) {
				double sum = 0;
				for (size_t i = 0; i < HIDENODE; i++) {
					sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
				}
				sum -= ::outLayer[j]->bias;

				::outLayer[j]->value = utils::sigmoid(sum);
			}

			// �������
			double error = 0.f;
			for (size_t i = 0; i < OUTNODE; i++) {
				// ���������� (y`-y) �ľ���ֵ  y����ѵ�����ݼ������(Ҳ���Ǳ��������������ݵĵ���������)
				double tmp = std::fabs(::outLayer[i]->value - idx.out[i]);
				error += (tmp * tmp) / 2.0;
			}

			// ʵʱ����������
			error_max = std::max(error_max, error);


			// *** ��ʼ���򴫲� *** 

			for (size_t i = 0; i < OUTNODE; i++) {
				double bias_delta = -(idx.out[i] - ::outLayer[i]->value) *
					::outLayer[i]->value * (1.0 - ::outLayer[i]->value);
				::outLayer[i]->bias_delta += bias_delta;
			}

			for (size_t i = 0; i < HIDENODE; i++) {
				for (size_t j = 0; j < OUTNODE; j++) {
					double weight_delta = (idx.out[j] - ::outLayer[j]->value) *
						::hideLayer[j]->value * (1.0 - ::outLayer[j]->value) *
						::hideLayer[i]->value;

					::hideLayer[i]->weight_delta[j] += weight_delta;
				}
			}

			for (size_t i = 0; i < HIDENODE; i++) {
				double sum = 0.f;
				for (size_t j = 0; j < OUTNODE; j++) {
					sum += -(idx.out[j] - ::outLayer[j]->value) *
						::hideLayer[j]->value * (1.0 - ::outLayer[j]->value) *
						::hideLayer[i]->weight[j];
				}
				::hideLayer[i]->bias_delta += sum * ::hideLayer[i]->value * (1.0 - ::hideLayer[i]->value);
			}

			for (size_t i = 0; i < INNODE; i++) {
				for (size_t j = 0; j < HIDENODE; j++) {
					double sum = 0.f;
					for (size_t k = 0; k < OUTNODE; k++) {
						sum += (idx.out[k] - ::outLayer[k]->value) *
							::outLayer[k]->value * (1.0 - ::outLayer[k]->value) *
							::hideLayer[j]->weight[k];
					}
					::inputLayer[i]->weight_delta[j] += sum * ::hideLayer[j]->value * (1.0 - ::hideLayer[j]->value) * ::inputLayer[i]->value;
				}

			}





			/*
				�۲����ĸ�ʽ�ӿ�֪�����Ƕ���һ���̶�����ɲ��֣����ǽ��ⲿ�ֳ�ȡΪ����
				Ҳ����: (y - y`) * y` * (1 - y`)
			*/
			//double attr_delta = 0.f;
			//for (size_t i = 0; i < OUTNODE; i++) {
			//	attr_delta += (idx.out[i] - ::outLayer[i]->value) *
			//		::outLayer[i]->value * (1.0 - ::outLayer[i]->value);
			//}






		}

		// ����ǰ������С���������õ����������������ѵ������Դﵽ����
		if (error_max < ::threshold) {
			std::cout << "Success with " << times + 1 << " times training" << std::endl;
			std::cout << "Maximum error: " << error_max << std::endl; // �����ǰ������
			break;
		}


		// û��ѵ���ɹ�
		// �ش�����Ȩֵ

		auto train_data_size = double(train_data.size());

		// ����㵽���ز��Ȩֵ
		for (size_t i = 0; i < INNODE; i++) {
			for (size_t j = 0; j < HIDENODE; j++) {
				::inputLayer[i]->weight[j] += rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
			}
		}

		// ���ز㵽������ƫ��ֵ��Ȩֵ
		for (size_t i = 0; i < HIDENODE; i++) {
			::hideLayer[i]->bias += rate * ::hideLayer[i]->bias_delta / train_data_size;
			for (size_t j = 0; j < OUTNODE; j++) {
				::hideLayer[i]->weight[j] += rate * ::hideLayer[i]->weight_delta[j] / train_data_size;
			}
		}

		// ������ƫ��ֵ
		for (size_t i = 0; i < OUTNODE; i++) {
			::outLayer[i]->bias += rate * ::outLayer[i]->bias_delta / train_data_size;
		}

	}

	// predicting
	std::vector<Sample> test_data = utils::getTestData("testdata.txt");

	for (auto& idx : test_data) {

		for (size_t i = 0; i < INNODE; i++) {
			::inputLayer[i]->value = idx.in[i];
		}

		// ��ʼ���򴫲�
		for (size_t j = 0; j < HIDENODE; j++) {
			double sum = 0;
			for (size_t i = 0; i < INNODE; i++) {
				sum += ::inputLayer[i]->value * inputLayer[i]->weight[j];
			}
			sum -= ::hideLayer[j]->bias;

			::hideLayer[j]->value = utils::sigmoid(sum);
		}

		for (size_t j = 0; j < OUTNODE; j++) {
			double sum = 0;
			for (size_t i = 0; i < HIDENODE; i++) {
				sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
			}
			sum -= ::outLayer[j]->bias;

			::outLayer[j]->value = utils::sigmoid(sum);

			idx.out.push_back(::outLayer[j]->value);


			// ���չʾ���
			for (auto& tmp : idx.in) {
				std::cout << tmp << " ";
			}
			for (auto& tmp : idx.out) {
				std::cout << tmp << " ";
			}
			std::cout << std::endl;
		}




	}



}