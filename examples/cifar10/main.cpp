
#include <iostream>
#include <string>
#include <vector>
#include <iosfwd>
#include <fstream>
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	const char *pLabel[] = 
	{
		"airplane\n",	"automobile\n",	"bird\n",	"cat\n",	"deer\n",
		"dog\n",		"frog\n",		"horse\n",	"ship\n",	"truck\n"
	};

	//读cifar二进制文件
	char data[2];
	uchar r[1024];
	uchar g[1024];
	uchar b[1024];

	memset(data, sizeof(data), 0);
	char file_name[128];

	FILE *fpLabel = fopen("E:\\Caffe\\CIFAR-10\\train label\\cifar-label.txt", "w+");

	int count = 1;
	for (int n = 1; n < 6; n++)
	{
		sprintf(file_name, "E:\\Caffe\\CIFAR-10\\cifar-10-batches-bin\\data_batch_%d.bin", n);

		size_t read_len;
		FILE *fp = NULL;
		fp = fopen(file_name, "rb");

		do 
		{
			read_len = fread(data, 1, 1, fp);
			if (read_len != 1)
				break;

			int num = data[0];
// 			cout << num << endl;
// 			cout << pLabel[num] << endl;
			fwrite(pLabel[num], 1, strlen(pLabel[num]), fpLabel);

			fread(r, 1, 1024, fp);
			fread(g, 1, 1024, fp);
			fread(b, 1, 1024, fp);

			//转换成jpg并保存
			Mat mat(32, 32, CV_8UC3);
			for (int i = 0; i < 32; i++)
			{
				for (int j = 0; j < 32; j++)
				{
					//for (int c = 0; c < 3; c++)
					{
						mat.at<Vec3b>(i, j)[0] = b[i * 32 + j];
						mat.at<Vec3b>(i, j)[1] = g[i * 32 + j];
						mat.at<Vec3b>(i, j)[2] = r[i * 32 + j];
					}
				}
			}

			char image_file[128];
			sprintf(image_file, "E:\\Caffe\\CIFAR-10\\train image\\%05d.jpg", count++);
			imwrite(image_file, mat);
		} while (1);

		cout << count << endl;
	}

	fclose(fpLabel);

	return 0;
}
