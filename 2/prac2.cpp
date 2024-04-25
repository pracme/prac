#include <bits/stdc++.h>
#include <iomanip>
#include <omp.h>

using namespace std;

void bubSort(vector<int> &arr){
int n=arr.size();
for(int i=0;i<n-1;++i){
    for(int j=0;j<n-i-1;++j){
        if(arr[j]>arr[j+1])
        {
            swap(arr[j],arr[j+1]);
        }
    }
}
}

void pbubSort(vector<int> &arr){
int n=arr.size();
for(int i=0;i<n-1;++i){
#pragma omp parallel for
    for(int j=0;j<n-i-1;++j){
        if(arr[j]>arr[j+1])
        {
            swap(arr[j],arr[j+1]);
        }
    }
}
}

void merge(vector<int> &arr, int left, int mid, int right){

  vector<int> L(arr.begin() + left, arr.begin() + mid + 1);
  vector<int> R(arr.begin() + mid + 1, arr.begin() + right + 1);
  
  int i = 0, j = 0, k = left;

  while (i < L.size() && j < R.size()) {
    arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
  }

  while (i < L.size()) arr[k++] = L[i++];
  while (j < R.size()) arr[k++] = R[j++];


}

void mergeSort(vector<int> &arr, int left,int right){
if(left<right){
    int mid=left+(right-left)/2;
    mergeSort(arr,left,mid);
    mergeSort(arr,mid+1,right);
    merge(arr,left,mid,right);
}
}

void pmergeSort(vector<int> &arr, int left,int right){
if(left<right){
    int mid=left+(right-left)/2;
    #pragma omp parallel sections
    {
    #pragma omp section    
    pmergeSort(arr,left,mid);

    #pragma omp section
    pmergeSort(arr,mid+1,right);
    }
    merge(arr,left,mid,right);
}
}

vector<int> genRD(int size){
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dis(1,999);

vector<int> arr(size);
  for (int i = 0; i < size; ++i)
  {
    arr[i] = dis(rd);
  }
  return arr;
}

void displayA(const vector<int> &arr){

for(int i=0;i<arr.size();i++){
cout<<arr[i]<<" ";
}

}


int main(){
    int size=0;
    cin>>size;
    vector<int> arr=genRD(size);
    vector<int> arrCopy = arr;
   displayA(arr);
    double starttime,endtime;


    auto bs=chrono::steady_clock::now();
    bubSort(arr);
    auto bse=chrono::steady_clock::now();
    chrono::duration<double> bst=bse-bs;
    cout<<" Time taken: "<<fixed<<setprecision(10)<<bst.count()<<endl;


    // auto bps=chrono::steady_clock::now();
    // pbubSort(arr);
    // auto bpse=chrono::steady_clock::now();
    // chrono::duration<double> bpst=bpse-bps;
    // cout<<" Time taken: "<<fixed<<setprecision(10)<<bpst.count()<<endl;

    starttime=omp_get_wtime();
    pbubSort(arr);
    endtime=omp_get_wtime();
       cout<<" Time taken: "<<fixed<<setprecision(7)<<endtime-starttime<<endl;

arr=arrCopy;

 auto mss=chrono::steady_clock::now();
    mergeSort(arr,0,size-1);
    auto mse=chrono::steady_clock::now();
    chrono::duration<double> mst=mse-mss;
    cout<<" Time taken: "<<fixed<<setprecision(10)<<mst.count()<<endl;

    // auto pms=chrono::steady_clock::now();
    // pmergeSort(arr,0,size-1);
    // auto pmse=chrono::steady_clock::now();
    // chrono::duration<double> pmst=pmse-pms;
    // cout<<" Time taken: "<<fixed<<setprecision(10)<<pmst.count()<<endl;

    starttime=omp_get_wtime();
    pmergeSort(arr,0,size-1);
    endtime=omp_get_wtime();
    cout<<" Time taken: "<<fixed<<setprecision(7)<<endtime-starttime<<endl;



displayA(arr);
    return 0;
}