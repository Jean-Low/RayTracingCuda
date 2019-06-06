#include <iostream>
#include "vec3.h"
#include "ray.h"

#include <fstream>
#include <cuda_profiler_api.h> //BASICALLY NOT WORKING TTxTT
#include <chrono>

using namespace std;

__device__
float hit_sphere(const vec3& center, float radious, const ray& r){
    vec3 oc = r.origin() - center;
    //baskara!
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc,oc) - radious*radious;
    float discriminant = b*b -4*a*c;

    if (discriminant < 0){
        return -1.0f;
    }
    else {
        return(-b-sqrt(discriminant)) / (2.0f*a);
    }

}

__device__
vec3 color(const ray& r){
    float hit = hit_sphere(vec3(0,0,-1),0.5f,r);

    if(hit > 0.0f){
        vec3 normal = unit_vector(r.point_at_parameter(hit) - vec3(0,0,-1));
        //some better color
        vec3 dif = normal - (unit_vector(r.direction()));
        float itensity = dif.length() - 1;
        //itensity = 1 - itensity;
        //printf("%f\n",dif.length());
        return vec3(0.2f * itensity, 0.8f * itensity, 0.5f * itensity);
        
    }
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f -t) *vec3(1.0f,1.0f,1.0f) + t*vec3(0.0f,0.0f,0.0f);
}

__global__ 
void renderPixel(int nx, int ny, int channels, int _outvec[],vec3 origin, vec3 l_l_corner, vec3 vertical, vec3 horizontal){

    //wich pixel am i seeing
    int index = blockIdx.x * blockDim.x + threadIdx.x; //(use multiplying it by channels)
    //get u and v
    int x = index % nx;
    int y = index / nx;
    float u = (float) x / float(nx);
    float v = (float) y / float(ny);
    //create ray
    ray r(origin, l_l_corner + u*horizontal + v*vertical);
    
    if((x % 1000 == 0) && (y % 1000 == 0)){
        printf("--- index - %i x-%i y-%i diry-%f u-%f v-%f\n",index,x,y,r.direction()[1],u,v);
        
    }
    
    //launch ray
    vec3 col = color(r);

    //write r g and b on outvec
    _outvec[index * channels + 0] = int(255.99f*col[0]);
    _outvec[index * channels + 1] = int(255.99f*col[1]);
    _outvec[index * channels + 2] = int(255.99f*col[2]);

}


int main() {
    //image resolution config
    int nx = 1920 * 2;
    int ny = 1080 * 2;
    int channels = 3;

    //camera options
    //field of view
    float fov = 12.0f;
    float fovRatio = (float)ny / (float)nx;
    vec3 horizontal(fov,0.0f,0.0f);
    vec3 vertical(0.0f,fov * fovRatio,0.0f);
    //bounderies and origin
    vec3 lower_left_corner(-fov/2,-fov * fovRatio/2,-1.0f);
    vec3 origin(0.0f,0.0f,0.0f);

    //threading options
    int blockSize = 64;

    //vector for output image
    int * outvec;// = new int[nx * ny * channels];

    //Start Cuda Profiler
    cout << "Starting GPU Parallelized section. \n";
    cudaProfilerStart();

    //Start Timer
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

    //mallocs for render
    cudaMallocManaged(&outvec, sizeof(int) * nx * ny * channels);
    
    //preparations and kernel call
    int numblocks = (nx * ny) / blockSize; //REMEMBER: each thread calculates a color for each channel, so the outvec is channelTimes bigger than the total number of threads
    int remainingThreads = (nx * ny) % blockSize;

    renderPixel<<<numblocks,blockSize>>>(nx,ny,channels,outvec,origin,lower_left_corner,vertical,horizontal);
    //make a especial block for the remaining pixels (in case that the total pixel size is not factored by blocksize)
    if(blockSize > 0){
        renderPixel<<<1,remainingThreads>>>(nx,ny,channels,outvec,origin,lower_left_corner,vertical,horizontal);
    }

    //sync everything
    cudaDeviceSynchronize();
    //End of GPU paralelization
    //End GPU time
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(end - start);

    //Start write timer
    chrono::high_resolution_clock::time_point timeStart = chrono::high_resolution_clock::now();

    //write file
    cout << "Writing File\n";
    ofstream image;
    image.open("image.ppm",ios::out);
    image << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = 0; i < nx * ny; i++){
        image << outvec[i * channels] << " " << outvec[i * channels + 1] << " " << outvec[i * channels + 2] << "\n";
    }
    image.close();

    //End Writer timer
    end = chrono::high_resolution_clock::now();

    //Flush timers
    cout << std::fixed << "GPU   time taken: " << time_span.count() << "\n";
    chrono::duration<double> write_time_span = chrono::duration_cast<chrono::duration<double>>(end - timeStart);
    cout << std::fixed << "Write time taken: " << write_time_span.count() << "\n";
    chrono::duration<double> total_time_span = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << std::fixed << "Total time taken: " << total_time_span.count() << "\n";
    
    //free memory
    cudaFree(outvec);
    
    //done
    cout << "DONE\n";
    cudaProfilerStop();

    return 0;
}

