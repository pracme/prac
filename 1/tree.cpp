#include<iostream>
#include<chrono>
#include<queue>
#include<omp.h>
using namespace std;
using namespace std::chrono;

typedef long long ll;


class TreeNode{
public:
	ll data;
	TreeNode* left;
	TreeNode* right;

	TreeNode(ll val){
		this->data = val;
		this->left = nullptr;
		this->right = nullptr;
	}
};


TreeNode* automatic_tree(ll N){
	TreeNode* root = new TreeNode(N);
	queue<TreeNode*> q;
	q.push(root);
	N--;
	while(!q.empty() && N>0){
		TreeNode* curr = q.front();
		q.pop();

		if(N>0){
			curr->left = new TreeNode(N);
			q.push(curr->left);
			N--;
		}
		if(N>0){
			curr->right = new TreeNode(N);
			q.push(curr->right);
			N--;
		}
	}
	return root;
}

void dfs_sequential(TreeNode* root,ll key, bool &flag){
	if(root == nullptr) return;

	if(root->data == key){
		flag = true;
		cout<<"\nkey found";
	}else{
		dfs_sequential(root->left,key,flag);
		dfs_sequential(root->right,key,flag);
	}
}

void dfs_parallel(TreeNode* root, ll key, bool flag){
	if(root==nullptr) return;
	if(flag) return;
	if(root->data= key){
		#pragma omp critical
		{

		flag = true;
		cout<<"\nkey found";
		}

	}else{
		#pragma omp parallel for
		for(int i=0; i<2;i++){
			if(i==0){
				dfs_parallel(root->left,key,flag);
			}else{
				dfs_parallel(root->right,key,flag);
			}
		}
	}
}

void bfs_sequential(TreeNode* root,ll key,bool &flag){
	queue<TreeNode*> q;
	q.push(root);
	while(!q.empty()){
		TreeNode* curr;
		ll size = q.size();
		for(ll i=0; i<size;i++){
			curr = q.front();
			q.pop();
			if(curr->data == key){
				flag = true;
			}
			if(curr->left){
				q.push(curr->left);
			}
			if(curr->right){
				q.push(curr->right);
			}
		}
	}
}

void bfs_parallel(TreeNode* root, ll key, bool &flag){
	queue<TreeNode*> q;
	q.push(root);

	while(!q.empty() && flag == false){
		TreeNode* curr;
		ll size = q.size();

		#pragma omp parallel for
		for(ll i=0; i<size;i++){
			curr = q.front();
			q.pop();
			if(curr->data = key){
				#pragma omp critical
				{
					flag = true;
				}
			}else if(!flag){
				#pragma omp critical
				{
					if(curr->right){

					q.push(curr->left);
					}
					if(curr->left){

					q.push(curr->right);
					}
				}
			}
		}
	}

}

int main(){
	int choice;
	cout<<"1>Automatic tree";
	cout<<"\n2> Manual Tree";
	cout<<"\nenter choice: ";
	cin>>choice;
	TreeNode* root;
	ll N;
	cout<<"\nEnter num of elements: ";
	cin>>N;
	if(choice==1){
		root = automatic_tree(N);
	}

	int search;
	cout<<"\n1>BFS search";
	cout<<"\n2>DFS search";
	cout<<"\nEnter search choice: ";
	cin>>search;

	if(search==1){
		ll key;
		cout<<"\nenter key to search: ";
		cin>>key;
		bool flag=false;
		auto start = high_resolution_clock::now();
		bfs_sequential(root,key,flag);
		auto end = high_resolution_clock::now();
		auto el = duration_cast<microseconds>(end-start);
		if(flag){
			cout<<"\n key found";
		}else{
			cout<<"\n key not found";
		}

		start = high_resolution_clock::now();
			flag = false;
		bfs_parallel(root,key,flag);
		end = high_resolution_clock::now();
		auto el_pl = duration_cast<microseconds>(end-start);
		cout<<"\n sequential time: "<<el.count()<<" ms";
		cout<<"\n parallel time: "<<el_pl.count()<<" ms";


	}else if(search==2){
		ll key;
		cout<<"\nenter key to search: ";
		cin>>key;
		bool flag;
		auto start = high_resolution_clock::now();
		dfs_sequential(root,key,flag);
		if(!flag){
			cout<<"\nkey not found";
		}

		auto end = high_resolution_clock::now();
		auto el = duration_cast<microseconds>(end-start);

		start = high_resolution_clock::now();
		flag = false;
		dfs_parallel(root,key,flag);
		end = high_resolution_clock::now();
		auto el_pl=duration_cast<microseconds>(end-start);

		cout<<"\nsequential time: "<<el.count()<<" ms";
		cout<<"\nparallel time: "<<el_pl.count()<<" ms";
	}

}
