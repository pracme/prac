#include <bits/stdc++.h>
#include <iomanip>
#include <omp.h>

using namespace std;

struct RD{
    double min_val,max_val,sum_val;
};

vector<double> genRD(int size){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(1.0,100.0);
    
    vector<double> data(size);
    for(int i=0;i<size;++i){
        data[i]=dis(rd);
        cout<<data[i]<<" ";
    }
    cout<<endl;
    return data;
}

RD seq(const vector<double> &arr){
    RD result={numeric_limits<double>::max(),numeric_limits<double>::min(),0.0};
    
    for(size_t i=0;i<arr.size();++i){
        result.min_val=min(result.min_val,arr[i]);
        result.max_val=max(result.max_val,arr[i]);
        result.sum_val+=arr[i];
    }
    
    return result;
}

#pragma omp reduction(redOP::RD::omp_out.min_val=min(omp_out.min_val,omp_in.min_val),omp_out.max_val=max(omp_out.max_val,omp_in.max_val),omp_out.sum_val+=omp_in.sum_val)

RD par(const vector<double> &arr){
    RD result={numeric_limits<double>::max(),numeric_limits<double>::lowest(),0.0};
    
#pragma omp parallel
    {
    RD lr=result;
#pragma omp for
    for(size_t i=0;i<arr.size();++i){
        lr.min_val=min(lr.min_val,arr[i]);
        lr.max_val=max(lr.max_val,arr[i]);
        lr.sum_val+=arr[i];
    }
        
#pragma omp critical
{
        result.min_val=min(result.min_val,lr.min_val);
        result.max_val=max(result.max_val,lr.max_val);
        result.sum_val+=lr.sum_val;  
}
    }
    return result;
}

int main(){
    int size=0;
    cin>>size;
    vector<double> data=genRD(size);
    
    auto s_s=chrono::steady_clock::now();
    RD sq=seq(data);
    auto s_e=chrono::steady_clock::now();
    chrono::duration<double> seq_time=s_e-s_s;
    double avgs=sq.sum_val/data.size();
    
    auto p_s=chrono::steady_clock::now();
    RD pr=par(data);
    auto p_e=chrono::steady_clock::now();
    chrono::duration<double> par_time=p_e-p_s;
    double avgp=pr.sum_val/data.size();
    
    cout<<"Seq \n min: "<<sq.min_val<<"\n "<<sq.max_val<<"\n "<<sq.sum_val<<"\n "<<avgs<<"\n"<<fixed<<setprecision(7)<<seq_time.count()<<endl;
    
     cout<<"Par \n min: "<<pr.min_val<<"\n "<<pr.max_val<<"\n "<<pr.sum_val<<"\n "<<avgp<<"\n"<<fixed<<setprecision(7)<<par_time.count()<<endl;

     return 0;
    
}