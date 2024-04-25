#include<iostream>
#include<vector>
#include<chrono>
#include<omp.h>
typedef long long ll;

using namespace std;

ll sequential_sum(vector<ll> nums){
	ll sum = 0;
	for(ll x: nums){
		sum += x;
	}
	return sum;
}

ll parallel_sum(vector<ll> nums){
	ll sum = 0;

	#pragma omp parallel for reduction(+:sum)
	for(ll i=0; i<nums.size();i++){
		sum += nums[i];
	}
	return sum;

}

ll sequential_max(vector<ll> nums){
	ll max = 0;
	for(ll x: nums){
		if(x > max){
			max = x;
		}
	}
	return max;
}

ll parallel_max(vector<ll> nums){
	ll max_num = nums[0];

	#pragma omp parallel for reduction(max:max_num)
	for(ll i =0; i < nums.size();i++){
		if(max_num < nums[i]){
			max_num = nums[i];
		}
	}
	return max_num;
}

ll sequential_min(vector<ll> nums){
	ll min = nums[0];
	for(ll x: nums){
		if(x < min){
			min = x;
		}
	}
	return min;
}
ll parallel_min(vector<ll> nums){
	ll min_num = nums[0];

	#pragma omp parallel for reduction(min:min_num)
	for(ll i=0; i<nums.size();i++){
		if(nums[i] < min_num){
			min_num = nums[i];
		}
	}
	return min_num;

}

ll sequential_avg(vector<ll> nums){
	ll avg = 0;
	for(ll x: nums){
		avg += x;
	}
	avg = avg/nums.size();
	return avg;
}

ll parallel_avg(vector<ll> nums){
	ll average = parallel_sum(nums);
	return average/nums.size();
}

int main(){
	int ch;
	ll N;
	vector<ll> nums;
	cout<<"Enter your choice:\n1>Automatic input generation\n2>input manually\n";
	cin>>ch;

	if(ch==1){
		cout<<"enter range: ";
		cin>>N;
		nums.resize(N);

		for(ll i=0; i<N;i++){
			nums[i] = i;
		}
		ll sum,min,max,avg;
		ll para_min;
		auto start = chrono::high_resolution_clock::now();
		min = sequential_avg(nums);
		auto end = chrono::high_resolution_clock::now();
		auto elapsed = chrono::duration_cast<chrono::microseconds>(end-start);


		start = chrono::high_resolution_clock::now();
		para_min = parallel_avg(nums);
		end = chrono::high_resolution_clock::now();
		auto parallel_elapsed = chrono::duration_cast<chrono::microseconds>(end-start);

		cout<<"\n------------------------------\n";
		cout<<"Operation\t"<<"Time taken(ns)\t\tResults";
		cout<<"\nmin:\t\t"<<elapsed.count()<<"\t\t\t"<<min;
		cout<<"\nparralel_min\t"<<parallel_elapsed.count()<<"\t\t\t"<<para_min;
	}
	return 0;
}

