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
vec3 color(const ray& r,const int worldsize, const vec3 _sphere_centers[], const vec3 _sphere_colors[], const float _sphere_radious[]){
    for(int i = 0; i < worldsize; i++){
        float hit = hit_sphere(_sphere_centers[i],_sphere_radious[i], r);

        if(hit > 0.0f){
            vec3 normal = unit_vector(r.point_at_parameter(hit) - _sphere_centers[i]);
            //some better color
            vec3 dif = normal - (unit_vector(r.direction()));
            float itensity = dif.length() - 1;
            //itensity = 1 - itensity;
            if(itensity < 0){
                itensity = 0.0f;
            }
            return _sphere_colors[i] * itensity;
            
        }
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f -t) *vec3(1.0f,1.0f,1.0f) + t*vec3(0.0f,0.0f,0.0f);
}

__device__
//return a "random" number betwen 1 and -1 based on 4 floats as seed (it is not random actually, but it is enough) 
float pseudo_rand(float a, float b, float c, float d){ 
    
    //a controlled chaotic expression
    float val = ( (((a + 134.854f + c)/3.3f) * c) / (d + 3.645f) );
    if(val == 0.0f){
        //if a or c is zero, use this other expression
        val = (c + d + 89.423f) * 9.308f * d * 1.54f + c; 
    }
    val *= 11.245f;

    //val = val % 2; //I cant use modulo on floats inside CUDA!!!
    //workaround:
    int precision = 100000; //how many decimal slots i want to keep (log(10), 5 in this case)
    int ret = (int) val % (2 * precision); //module it with some precision
    val = (float)ret / (precision); // make ret a floating point
    
    return (val - 1.0f);
}

__global__ 
void renderPixel(int nx, int ny, int channels, int antia, int _outvec[], vec3 _sphere_centers[] ,vec3 _sphere_colors[], float _sphere_radious[],int worldsize, vec3 origin, vec3 l_l_corner, vec3 vertical, vec3 horizontal){

    //wich pixel am i seeing
    int index = blockIdx.x * blockDim.x + threadIdx.x; //(use multiplying it by channels)
    //get u and v
    int x = index % nx;
    int y = index / nx;
    float u = (float) x / float(nx);
    float v = (float) y / float(ny);
    //create ray
    ray r(origin, l_l_corner + u*horizontal + v*vertical);

    //info for antialiasing
    float difx = (((float) x + 1) / float(nx)) - u; //diference betwen this u and nexts pixel u
    float dify = (((float) y + 1) / float(ny)) - v; //diference betwen this v and nexts pixel v
    float udif,vdif = 0.0f;

    //launch ray for each antialiasing ray
    vec3 col = vec3(0,0,0);
    udif = pseudo_rand((float)index, u, (float)nx * 5.1f * (y), (float) x / y) * difx ;
    vdif = pseudo_rand((float)index, v, (float)ny * 1.52f * (x), (float) y / x) * dify;
    for(int i = 0; i < antia; i ++){
        col += color(r,worldsize,_sphere_centers,_sphere_colors,_sphere_radious);
        udif -=  pseudo_rand((float)index, u, (float)nx * 5.1f * i * (y), (float) x / y) * difx;
        vdif -=  pseudo_rand((float)index, v, (float)ny * 1.52f * i * (x), (float) y / x) * dify;
        //printf("u%f, v%f, i-%i\n",udif,vdif,i);
        //change ray so it is a little diferent but inside the same pixel bounderies
        r.B = l_l_corner + ((u + udif)*horizontal + (v + vdif)*vertical);
        
    }
    col /= antia;
    

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

    //amount of antialiasing
    int antia = 8;
    
    //Set var for world inside cuda
    vec3 *sphere_centers;
    vec3 *sphere_colors;
    float *sphere_radious;
    //list for storing world values (for readbility)
    int worldsize = 3;

    //camera options
    //field of view
    float fov = 4.0f;
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
    //also for world
    cudaMallocManaged(&sphere_centers, sizeof(vec3) * worldsize);
    cudaMallocManaged(&sphere_colors, sizeof(vec3) * worldsize);
    cudaMallocManaged(&sphere_radious, sizeof(float) * worldsize);

    sphere_centers[0] =  vec3(0,1,-2);
    sphere_colors [0] =  vec3(0.2f, 0.8f, 0.5f);
    sphere_radious[0] =  0.5f;
    sphere_centers[1] =  vec3(1,-0.5f,-2);
    sphere_colors [1] =  vec3(0.8f, 0.2f, 0.1f);
    sphere_radious[1] =  0.5f;
    sphere_centers[2] =  vec3(-1,-0.5f,-2);
    sphere_colors [2] =  vec3(0.9f, 0.8f, 0.4f);
    sphere_radious[2] =  0.5f;
    
    //preparations and kernel call
    int numblocks = (nx * ny) / blockSize; //REMEMBER: each thread calculates a color for each channel, so the outvec is channelTimes bigger than the total number of threads
    int remainingThreads = (nx * ny) % blockSize;

    renderPixel<<<numblocks,blockSize>>>(nx,ny,channels,antia,outvec,sphere_centers,sphere_colors,sphere_radious,worldsize,origin,lower_left_corner,vertical,horizontal);
    //make a especial block for the remaining pixels (in case that the total pixel size is not factored by blocksize)
    if(blockSize > 0){
        renderPixel<<<1,remainingThreads>>>(nx,ny,channels,antia,outvec,sphere_centers,sphere_colors,sphere_radious,worldsize,origin,lower_left_corner,vertical,horizontal);
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
    cudaFree(sphere_centers);
    cudaFree(sphere_colors);
    cudaFree(sphere_radious);
    
    //done
    cout << "DONE\n";
    cudaProfilerStop();

    return 0;
}

