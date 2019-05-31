#include <iostream>
#include <fstream>

using namespace std;

__global__ 
void renderPixel(int nx, int ny, int channels, int _outvec[]){

    //wich pixel am i seeing
    int index = blockIdx.x * blockDim.x + threadIdx.x; //(use multiplying it by channels)
    //change r g and b
    float r = ((float)index / (nx * ny)) * 0.95 ;
    float g = (1.0 - ((float)index / (nx * ny))) * 0.95;
    float b = 0.1;
    //write r g and b on outvec
    _outvec[index * channels] = int(255.99*r);
    _outvec[index * channels + 1] = int(255.99*g);
    _outvec[index * channels + 2] = int(255.99*b);

}


int main() {
    //image resolution config
    int nx = 800;
    int ny = 200;
    int channels = 3;

    //threading options
    int blockSize = 64;

    //vector for output image
    int * outvec;// = new int[nx * ny * channels];

    //mallocs for render
    cudaMallocManaged(&outvec, sizeof(int) * nx * ny * channels);
    
    cout << "PRAY\n";
    //preparations and kernel call
    int numblocks = (nx * ny) / blockSize; //REMEMBER: each thread calculates a color for each channel, so the outvec is channelTimes bigger than the total number of threads
    int remainingThreads = (nx * ny) % blockSize;

    renderPixel<<<numblocks,blockSize>>>(nx,ny,channels,outvec);
    //make a especial block for the remaining pixels (in case that the total pixel size is not factored by blocksize)
    if(blockSize > 0){
        renderPixel<<<1,remainingThreads>>>(nx,ny,channels,outvec);
    }

    //sync everything
    cudaDeviceSynchronize();

    //write file
    cout << "Writing\n";
    ofstream image;
    image.open("image.ppm",ios::out);
    image << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = 0; i < nx * ny; i++){
        image << outvec[i * channels] << " " << outvec[i * channels + 1] << " " << outvec[i * channels + 2] << "\n";
    }
    image.close();
    
    //free memory
    cudaFree(outvec);
    //done
    cout << "DONE\n";

    return 0;
}

