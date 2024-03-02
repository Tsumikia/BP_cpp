#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <random>


#define INNODE 2
#define HIDENODE 4
#define OUTNODE 1

double rate = 0.8; // 步长
double threshold = 1e-4; // 允许的最大误差
size_t mosttimes = 1e7; // 最大迭代次数


struct Sample { // 样本
	std::vector<double> in, out;

};

struct Node { // 单个神经元模型
	double value{}, bias{}, bias_delta{};
	std::vector<double> weight, weight_delta; // 权值
};

namespace utils {

	// 激活函数
	inline double sigmoid(double x) {
		double res = 1.0 / (1.0 + std::exp(-x));
		return res;
	}

	std::vector<double> getFileData(std::string filename) {
		std::vector<double> res;

		std::ifstream in(filename); // 文件输入流
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

// 输入层， 隐藏层， 输出层
Node* inputLayer[INNODE], * hideLayer[HIDENODE], * outLayer[OUTNODE];

inline void init() {
	std::mt19937 rd;
	rd.seed(std::random_device()()); // 随机种子并调用operator()以获取随机数

	std::uniform_real_distribution<double> distribution(-1, 1);  // -1 到 1.0 的随机实数

	for (size_t i = 0; i < INNODE; i++) {
		::inputLayer[i] = new Node(); // 初始化INNODE个节点
		for (size_t j = 0; j < HIDENODE; j++) {
			::inputLayer[i]->weight.push_back(distribution(rd)); // 对应的是第i个节点到第j个隐藏层的weight
			::inputLayer[i]->weight_delta.push_back(0.f); // 修正值一开始都是0
		}
	}

	// 以下为对隐藏层的初始化
	for (size_t i = 0; i < HIDENODE; i++) {
		::hideLayer[i] = new Node();
		::hideLayer[i]->bias = distribution(rd);
		for (size_t j = 0; j < OUTNODE; j++) {
			::hideLayer[i]->weight.push_back(distribution(rd));
			::hideLayer[i]->weight_delta.push_back(0.f);
		}
	}

	// 输出层
	for (size_t i = 0; i < OUTNODE; i++) {
		::outLayer[i] = new Node();
		::outLayer[i]->bias = distribution(rd);
	}
}


/*
	重置了神经网络每个节点的"可以动态调整的调整值"
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
		// 每次迭代前先重置delta值
		reset_delta();

		// 最大误差
		double error_max = 0.f;


		// 对于每一个训练样本
		for (auto& idx : train_data) {
			// 输入层初始化
			for (size_t i = 0; i < INNODE; i++) {
				::inputLayer[i]->value = idx.in[i];
			}

			// *** 正向传播 ***
			for (size_t j = 0; j < HIDENODE; j++) {
				double sum = 0; // 求多项式的和
				for (size_t i = 0; i < INNODE; i++) {
					// 第i个节点的值 乘上 第i个节点到第j个节点的权值
					sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
				}
				sum -= ::hideLayer[j]->bias; // 还要减掉第j个隐藏层节点的偏置值

				// 然后第j个隐藏层的值就已经确定
				// 将多项式求和结果带入激活函数
				::hideLayer[j]->value = utils::sigmoid(sum);
			}

			// 生成输出层的value
			for (size_t j = 0; j < OUTNODE; j++) {
				double sum = 0;
				for (size_t i = 0; i < HIDENODE; i++) {
					sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
				}
				sum -= ::outLayer[j]->bias;

				::outLayer[j]->value = utils::sigmoid(sum);
			}

			// 计算误差
			double error = 0.f;
			for (size_t i = 0; i < OUTNODE; i++) {
				// 这里是生成 (y`-y) 的绝对值  y就是训练数据集的输出(也就是本案例中输入数据的第三个数据)
				double tmp = std::fabs(::outLayer[i]->value - idx.out[i]);
				error += (tmp * tmp) / 2.0;
			}

			// 实时更新最大误差
			error_max = std::max(error_max, error);


			// *** 开始反向传播 *** 

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
				观察那四个式子可知，他们都有一个固定的组成部分，于是将这部分抽取为常量
				也就是: (y - y`) * y` * (1 - y`)
			*/
			//double attr_delta = 0.f;
			//for (size_t i = 0; i < OUTNODE; i++) {
			//	attr_delta += (idx.out[i] - ::outLayer[i]->value) *
			//		::outLayer[i]->value * (1.0 - ::outLayer[i]->value);
			//}






		}

		// 当当前最大误差小于我们设置的允许最大误差，即表明训练结果以达到良好
		if (error_max < ::threshold) {
			std::cout << "Success with " << times + 1 << " times training" << std::endl;
			std::cout << "Maximum error: " << error_max << std::endl; // 输出当前最大误差
			break;
		}


		// 没有训练成功
		// 回传调整权值

		auto train_data_size = double(train_data.size());

		// 输入层到隐藏层的权值
		for (size_t i = 0; i < INNODE; i++) {
			for (size_t j = 0; j < HIDENODE; j++) {
				::inputLayer[i]->weight[j] += rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
			}
		}

		// 隐藏层到输出层的偏置值和权值
		for (size_t i = 0; i < HIDENODE; i++) {
			::hideLayer[i]->bias += rate * ::hideLayer[i]->bias_delta / train_data_size;
			for (size_t j = 0; j < OUTNODE; j++) {
				::hideLayer[i]->weight[j] += rate * ::hideLayer[i]->weight_delta[j] / train_data_size;
			}
		}

		// 输出层的偏置值
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

		// 开始正向传播
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


			// 输出展示结果
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