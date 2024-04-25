#include <bits/stdc++.h>
#include <iomanip>
#include <omp.h>

using namespace std;

vector<int> genRD(int size){
random_device rd;
mt19937 gen(rd());
uniform_int_distribution dis(1,99999);

vector<int> data(size);

for(int i=0;i<size;++i){
    data[i]=dis(rd);

}
return data;
}

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

while(i<L.size() && j<R.size()){
    arr[k++]=(L[i]<=R[j])?L[i++]:R[j++];
}

while(i<L.size()) arr[k++]=L[i++];
while(j<R.size()) arr[k++]=R[j++];

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
    mSort(arr,left,mid+1);
    #pragma omp section
    mSort(arr,mid+1,right);
    }
    merge(arr,left,mid,right);
}
}

void disArr(vector<int> &arr){
    for(int i=0;i<arr.size();++i){
        cout<<arr[i]<<" ";
    }
}

int main(){
    int size=0;
    cin>>size;
    vector<int> arr=genRD(size);
    disArr(arr);
    vector <int> arrcpy=arr;
    double startime,endtime;

    startime=omp_get_wtime();
    bSort(arr);
    endtime=omp_get_wtime();
    cout<<"bsort time: "<<endtime - startime<<endl;

    auto bs=chrono::steady_clock::now();
    pbSort(arr);
    auto bse=chrono::steady_clock::now();
    chrono::duration<double> bst=bse-bs;
    cout<<" pbsort Time taken: "<<fixed<<setprecision(10)<<bst.count()<<endl;


arr=arrcpy;

    startime=omp_get_wtime();
    mSort(arr,0,arr.size()-1);
    endtime=omp_get_wtime();
    cout<<"msort time: "<<endtime - startime<<endl;
    
    startime=omp_get_wtime();
    pmSort(arr,0,arr.size()-1);
    endtime=omp_get_wtime();
    cout<<"pmsort time: "<<endtime - startime<<endl;


disArr(arr);
    return 0;
}