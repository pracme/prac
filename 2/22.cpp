#include <bits/stdc++.h>
#include <iomanip>
#include <omp.h>

using namespace std;

void bSort(vector<int> &arr){
  int n=arr.size();

  for(int i=0;i<n-1;++i){
    for(int j=0;j<n-i-1;++j){
        if(arr[j]>arr[j+1]){
            swap(arr[j],arr[j+1]);
        }

    }

  }  
}

void pbSort(vector<int> &arr){
  int n=arr.size();

  for(int i=0;i<n-1;++i){
    #pragma omp parallel for
    for(int j=0;j<n-i-1;++j){
        if(arr[j]>arr[j+1]){
            swap(arr[j],arr[j+1]);
        }

    }

  }  
}

void merge(vector<int> &arr,int left,int mid,int right){
vector<int> L(arr.begin()+left,arr.begin()+mid+1);
vector<int> R(arr.begin()+mid+1,arr.begin()+right+1);

int i=0,j=0,k=left;

while(i<L.size()&& j<R.size()){
    arr[k++]=(L[i]<=R[j])?L[i++]:R[j++];
}
while(i<L.size())arr[k++]=L[i++];
while(j<R.size())arr[k++]=R[j++];

}

void mSort(vector<int> &arr,int left,int right){

if(left<right){
int mid=left+(right-left)/2;
mSort(arr,left,mid);
mSort(arr,mid+1,right);
merge(arr,left,mid,right);
}
}

void pmSort(vector<int> &arr,int left,int right){
if(left<right){
int mid=left+(right-left)/2;
#pragma omp parallel sections
{
#pragma omp section
mSort(arr,left,mid);
#pragma omp section
mSort(arr,mid+1,right);
}
merge(arr,left,mid,right);
}
}

vector<int> genRD(int size){
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> dis(1,999);

vector<int> data(size);

for(int i=0;i<size;++i){
    data[i]=dis(rd);
}
return data;

}

void displayArray(vector<int> &arr){

for(int i=0;i<arr.size();i++){
cout<<arr[i]<<" ";
}

}

int main(){
    int size=0;
    cin>>size;
    vector<int> arr=genRD(size);
    displayArray(arr);
    vector<int> arrcpy=arr;
    double starttime,endtime;


    auto bst=chrono::steady_clock::now();
    bSort(arr);
    auto bet=chrono::steady_clock::now();
    chrono::duration<double> btime=bet-bst;
    cout<<" Time taken: "<<fixed<<setprecision(10)<<btime.count()<<endl;

    starttime=omp_get_wtime();
    pbSort(arr);
    endtime=omp_get_wtime();
    cout<<" Time taken: "<<fixed<<setprecision(10)<<endtime-starttime<<endl;

    arr=arrcpy;

    starttime=omp_get_wtime();
    mSort(arr,0,size-1);
    endtime=omp_get_wtime();
    cout<<" Time taken: "<<fixed<<setprecision(10)<<endtime-starttime<<endl;

    starttime=omp_get_wtime();
    pmSort(arr,0,size-1);
    endtime=omp_get_wtime();
    cout<<" Time taken: "<<fixed<<setprecision(10)<<endtime-starttime<<endl;

displayArray(arr);


    return 0;
}