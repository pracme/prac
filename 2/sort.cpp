#include<iostream>
#include<vector>
#include<chrono>
#include<omp.h>
#define CUTT_OFF 1000
using namespace std;
using namespace std::chrono;
typedef long long ll;

void print_array(vector<ll>nums){
	cout<<"\n[ ";
	for(ll x: nums){
		cout<<x<<", ";
	}
	cout<<" ]";
}

void bubble_sort_sequential(vector<ll> &nums){

	for(int i=0; i<nums.size()-1;i++){

		for(int j=0; j<nums.size()-i-1;j++){
			if(nums[j] > nums[j+1]){
				swap(nums[j],nums[j+1]);
			}
		}
	}
}

void bubble_sort_parallel(vector<ll> &nums){
	ll size = nums.size();
	for(ll i=0; i<size;i++){
		ll first = i%2;

		#pragma omp parallel for shared(nums,first)
		for(ll j = first; j<size-1;j+=2){
			swap(nums[j],nums[j+1]);
		}
	}
}

void merge(vector<ll>left,vector<ll>right,vector<ll>&nums){
	ll l=0;
	ll r = 0;
	ll k =0;
	ll l_size = left.size();
	ll r_size = right.size();

	while(l < l_size && r< r_size){
		if(left[l]<=right[r]){
			nums[k] = left[l];
			l++;
			k++;
		}else{
			nums[k] = right[r];
			r++;
			k++;
		}
	}
	while(l<l_size){
		nums[k] = left[l];
		l++;
		k++;
	}
	while(r<r_size){
		nums[k] = right[r];
		r++;
		k++;
	}
}

void merge_sort_sequential(vector<ll> &nums){
	ll size = nums.size();
	if(size <= 1) return;
	ll mid = size/2;

	vector<ll>left;
	vector<ll>right;

	for(ll i=0; i<size;i++){
		if(i < mid){
			left.push_back(nums[i]);
		}else{
			right.push_back(nums[i]);
		}
	}
	merge_sort_sequential(left);
	merge_sort_sequential(right);
	merge(left,right,nums);
}

void merge_sort_parallel(vector<ll> &nums){
	ll size = nums.size();


	if(size<=CUTT_OFF){
		merge_sort_sequential(nums);
	}else{

	ll mid = size/2;

	vector<ll>left;
	vector<ll>right;

	for(ll i=0; i<size;i++){
		if(i<mid){
			left.push_back(nums[i]);
		}else{
			right.push_back(nums[i]);
		}
	}
	#pragma omp parallel
	{
		#pragma omp single nowait
		{
			#pragma omp task
			merge_sort_parallel(left);

			#pragma omp task
			merge_sort_parallel(right);

		}
	}
	merge(left,right,nums);
	}
}

int main(){
	int choice;
	cout<<"1>automatic generation";
	cout<<"\n2>manual generation";
	cout<<"\nchoose input method: ";
	cin>>choice;

	vector<ll> nums;
	ll size;
	cout<<"\nenter range: ";
	cin>>size;


	if(choice==1){
		nums.resize(size);
		for(ll i=0; i<size;i++){
			nums[i] = size-i;
		}
	}else if (choice == 2){
		nums.resize(size);
		for(ll i=0; i<size;i++){
			cin>>nums[i];
		}
	}
	vector<ll> copy_nums = nums;

	int sort_choice;
	cout<<"\n1>bubble sort";
	cout<<"\n2>merge sort";
	cout<<"\nenter choice: ";
	cin>>sort_choice;

	if(sort_choice==1){
		auto start = high_resolution_clock::now();
		bubble_sort_sequential(nums);
		auto end = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(end-start);

		start = high_resolution_clock::now();
		bubble_sort_parallel(copy_nums);
		end = high_resolution_clock::now();
		auto elapsed_parallel = duration_cast<milliseconds>(end-start);
		print_array(copy_nums);
		cout<<"\n------------------------------------";
		cout<<"\nalgo\t\t"<<"time taken(ms)";
		cout<<"\nsequential\t"<<elapsed.count();
		cout<<"\nparallel\t"<<elapsed_parallel.count();

	}else if(sort_choice==2){
		// cout<<"\nrun";
		auto start = high_resolution_clock::now();
		merge_sort_sequential(nums);
		auto end = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(end-start);

		start = high_resolution_clock::now();
		merge_sort_parallel(copy_nums);
		end = high_resolution_clock::now();
		auto elapsed_parallel = duration_cast<milliseconds>(end-start);
		print_array(copy_nums);
		cout<<"\n------------------------------------";
		cout<<"\nalgo\t\t"<<"time taken(ms)";
		cout<<"\nsequential\t"<<elapsed.count();
		cout<<"\nparallel\t"<<elapsed_parallel.count();
	}



}
