#include <iostream>
#include <queue>
#include <omp.h>
#include <chrono>
#include <vector>
using namespace std;

#define COUNT 10 // Define COUNT for setting the width of the tree when printing

class Node // Node class for the binary tree
{
public:
  int data;           // Data stored in the node
  Node *left, *right; // Pointers to left and right child nodes

  // Constructor to initialize a node with given data
  Node(int data)
  {
    this->data = data;  // Assign data to the node
    this->left = NULL;  // Initialize left child pointer as NULL
    this->right = NULL; // Initialize right child pointer as NULL
  }
};

void print2DUtil(Node *root, int space) // Function to print 2D representation of the binary tree
{
  // Base case: If the root is NULL, return
  if (root == NULL)
    return;

  // Increase the space between levels
  space += COUNT;

  // Process right child first (to print rightmost nodes first)
  print2DUtil(root->right, space);

  // Print current node after appropriate spacing
  cout << endl;
  for (int i = COUNT; i < space; i++)
    cout << " ";              // Print spaces to maintain proper indentation
  cout << root->data << "\n"; // Print the data of the current node

  // Process left child
  print2DUtil(root->left, space);
}

void print2D(Node *root) // Function to print 2D representation of the binary tree, starting from the root node
{
  // Call the utility function print2DUtil with the root node and initial indentation level 0
  print2DUtil(root, 0);
}

// Function to perform sequential breadth-first search (BFS) on a binary tree
void sequentialBFS(Node *root)
{
  // Create a queue to store nodes for BFS traversal
  queue<Node *> q;
  // Enqueue the root node to start BFS traversal
  q.push(root);

  // Continue BFS traversal until the queue is empty
  while (!q.empty())
  {
    // Dequeue the current node
    Node *current = q.front();
    q.pop();

    // Print the data of the current node
    cout << current->data << " ";

    // Enqueue the left child if it exists
    if (current->left != NULL)
    {
      q.push(current->left);
    }
    // Enqueue the right child if it exists
    if (current->right != NULL)
    {
      q.push(current->right);
    }
  }
}

// Function to perform parallel breadth-first search (BFS) on a binary tree
void parallelBFS(Node *root)
{
  // Create a queue to store nodes for BFS traversal
  queue<Node *> q;
  // Enqueue the root node to start BFS traversal
  q.push(root);

  // Start parallel region
#pragma omp parallel shared(q)
  {
    // Execute the following block of code only once in parallel
#pragma omp single nowait
    {
      // Continue BFS traversal until the queue is empty
      while (!q.empty())
      {
        // Get the number of nodes at the current level
        int q_size = q.size();

        // Wait for all tasks in the current level to complete before proceeding
#pragma omp taskwait
        // Process each node at the current level
        for (int i = 0; i < q_size; i++)
        {
          // Dequeue the current node
          Node *current = q.front();
          q.pop();

          // Print the data of the current node (critical section to ensure thread safety)
#pragma omp critical
          {
            cout << current->data << " ";
          }

          // Enqueue the left child if it exists
          if (current->left != NULL)
          {
            q.push(current->left);
          }
          // Enqueue the right child if it exists
          if (current->right != NULL)
          {
            q.push(current->right);
          }
        }
      }
    }
  }
}

// Function to perform sequential depth-first search (DFS) on a binary tree
void sequentialDFS(Node *root)
{
  // Base case: If the root is NULL, return (no further traversal)
  if (root == NULL)
    return;

  // Print the data of the current node
  cout << root->data << " ";

  // Recursively traverse the left subtree
  sequentialDFS(root->left);

  // Recursively traverse the right subtree
  sequentialDFS(root->right);
}

// Function to perform parallel depth-first search (DFS) on a binary tree
void parallelDFSUtil(Node *node, int level)
{
  // Base case: If the current node is NULL, return (no further traversal)
  if (node == NULL)
    return;

    // Create a task to process the current node
#pragma omp task shared(node, level)
  {
    // Print the data of the current node
    cout << node->data << " ";

    // Increment the level for the next recursive calls
    int leftLevel = level + 1;

    // Traverse the left subtree in parallel
    parallelDFSUtil(node->left, leftLevel);

    // Wait for all tasks spawned by traversing the left subtree to complete
#pragma omp taskwait

    // Traverse the right subtree in parallel
    parallelDFSUtil(node->right, leftLevel);
  }
}

// Function to initiate parallel DFS traversal from the root of the binary tree
void parallelDFS(Node *root)
{
  // Initialize the level of the root node
  int rootLevel = 0;

  // Start the parallel region
#pragma omp parallel
  {
    // Execute the following block only once in parallel
#pragma omp single nowait
    {
      // Initiate parallel DFS traversal from the root node
      parallelDFSUtil(root, rootLevel);
    }
  }
}

// Preorde, Inorder, Postorder

void preorder(Node *root)
{
  if (root == NULL)
    return;

  cout << root->data << " ";
  preorder(root->left);
  preorder(root->right);
}

void postorder(Node *root)
{
  if (root == NULL)
    return;

  postorder(root->left);
  postorder(root->right);
  cout << root->data << " ";
}

void inorder(Node *root)
{
  if (root == NULL)
    return;

  inorder(root->left);
  cout << root->data << " ";
  inorder(root->right);
}

// Function to build a binary tree based on user input
Node *buildTree()
{
  int value;
  cout << "\n ***** [ HPC ASSIGNMENT 1 ] ***** \n --------------------------------\n\n Building Binary Tree..... \n\n-> Enter Root Value of Tree: ";
  cin >> value;

  // Create the root node with the specified value
  Node *root = new Node(value);

  // Create a queue to keep track of nodes whose children need to be set
  queue<Node *> q;
  q.push(root);

  // Continue until all nodes are processed
  while (!q.empty())
  {
    // Dequeue the current node
    Node *current = q.front();
    q.pop();
    int leftValue, rightValue;
    // Prompt the user to enter the value of the left child of the current node
    cout << "-> Enter Left Child of " << current->data << " (-1 for No Child, -2 to Skip Rest): ";
    cin >> leftValue;

    // If the user inputs -2, skip the remaining nodes
    if (leftValue == -2)
    {
      while (!q.empty())
      {
        q.pop();
      }
      break;
    }

    // If the user inputs a value other than -1, create the left child node and enqueue it
    if (leftValue != -1)
    {
      current->left = new Node(leftValue);
      q.push(current->left);
    }

    // Prompt the user to enter the value of the right child of the current node
    cout << "-> Enter Right Child of " << current->data << " (-1 for no Child, -2 to Skip Rest): ";
    cin >> rightValue;

    // If the user inputs -2, skip the remaining nodes
    if (rightValue == -2)
    {
      while (!q.empty())
      {
        q.pop();
      }
      break;
    }
    // If the user inputs a value other than -1, create the right child node and enqueue it
    if (rightValue != -1)
    {
      current->right = new Node(rightValue);
      q.push(current->right);
    }
  }
  // Return the root of the constructed binary tree
  return root;
}
// Call the buildTree function to construct the binary tree and assign its root to the 'root' pointer
Node *root = buildTree();

// Function to measure the execution time of a given function
void measureTime(void (*func)(Node *), string name)
{
  // Record the start time before executing the provided function
  auto start = chrono::high_resolution_clock::now();

  // Execute the provided function
  func(root);

  // Record the stop time after executing the provided function
  auto stop = chrono::high_resolution_clock::now();

  // Calculate the duration of the execution
  auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

  // Print the name of the function and its execution time
  cout << name << " Time Taken is: " << duration.count() << " microseconds" << endl;
}

// MAIN-FUNCTION
int main()
{
  cout << "\n ***** Constructed BINARY TREE ***** \n ------------------------------------"
       << endl;
  print2D(root);

  cout << "\n\n-> Preorder Traversal: ";
  preorder(root);
  cout << "\n-> Postorder Traversal: ";
  postorder(root);
  cout << "\n-> Inorder Traversal: ";
  inorder(root);

  cout << "\n\n-> Sequential Breadth-First Search(BFS): ";
  measureTime(sequentialBFS, "-->");
  cout << "-> Parallel Breadth-First Search(BFS): ";
  measureTime(parallelBFS, "-->");
  cout << "\n-> Sequential Depth-First Search(DFS): ";
  measureTime(sequentialDFS, "-->");
  cout << "-> Parallel Depth-First Search(DFS): ";
  measureTime(parallelDFS, "-->");
  cout << " " << endl;
  return 0;
}
