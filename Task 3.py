#1.Recursive Factorial

def fact(n):
    if(n<=1):
        return 1
    else:
        return n*fact(n-1)
    
factnumber=int(input("Enter a number: "))
factnumber=fact(factnumber)
print(factnumber)



#2.Palindrome Linked List

class Node:
    def __init__(self, data):
        self.data=data
        self.next=None

class LinkedList:
    def __init__(self):
        self.head=None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head=new_node
            return
        last_node=self.head
        while last_node.next:
            last_node=last_node.next
        last_node.next=new_node

    def is_palindrome(self):
        elements=[]
        current_node = self.head
        while current_node:
            elements.append(current_node.data)
            current_node=current_node.next
        return elements==elements[::-1]
    
my_list=LinkedList()
my_list.append(int(input("Enter 1st Number: ")))
my_list.append(int(input("Enter 2nd Number: ")))
my_list.append(int(input("Enter 3rd Number: ")))
my_list.append(int(input("Enter 4th Number: ")))
my_list.append(int(input("Enter 5th Number: ")))

if my_list.is_palindrome():
    print("The Linked List is a palindrome.")
else:
    print("The Linked List is not a palindrome.")



#3.Merge Sorted Arrays

import heapq

def merge_sorted_array(arr1,arr2):
    merged=[]
    for element in heapq.merge(arr1,arr2):
        merged.append(element)
    return merged
    
arr1=[1,3,5]
arr2=[2,4,6]

merged_array=merge_sorted_array(arr1,arr2)
print(merged_array)



#4.Binary Search Tree

class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, root, key):
        if root is None:
            return TreeNode(key)
        if key < root.key:
            root.left = self._insert_recursive(root.left, key)
        elif key > root.key:
            root.right = self._insert_recursive(root.right, key)
        return root

    def search(self, key):
        return self._search_recursive(self.root, key)

    def _search_recursive(self, root, key):
        if root is None or root.key == key:
            return root
        if key < root.key:
            return self._search_recursive(root.left, key)
        return self._search_recursive(root.right, key)

    def delete(self, key):
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, root, key):
        if root is None:
            return root

        if key < root.key:
            root.left = self._delete_recursive(root.left, key)
        elif key > root.key:
            root.right = self._delete_recursive(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp

            temp = self._min_value_node(root.right)
            root.key = temp.key
            root.right = self._delete_recursive(root.right, temp.key)

        return root

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder_traversal(self, root):
        result = []
        if root:
            result += self.inorder_traversal(root.left)
            result.append(root.key)
            result += self.inorder_traversal(root.right)
        return result

bst = BinarySearchTree()
bst.insert(50)
bst.insert(30)
bst.insert(20)
bst.insert(40)
bst.insert(70)
bst.insert(60)
bst.insert(80)

print("Inorder traversal:", bst.inorder_traversal(bst.root))

search_key = 40
result = bst.search(search_key)
if result:
    print




#5.Longest Palindromic Substring

def long_palindrome(s):
    if not s:
        return ""
    
    def left_right(left,right):
        while left>=0 and right<len(s) and s[left]==s[right]:
            left -=1
            right +=1
        return s[left+1:right]
    
    longest_palindrome=""
    for i in range(len(s)):
        palindrome_odd=left_right(i,i)
        if len(palindrome_odd)>len(longest_palindrome):
            longest_palindrome=palindrome_odd

        palindrome_even=left_right(i,i+1)
        if len(palindrome_even)>len(longest_palindrome):
            longest_palindrome=palindrome_even
    return longest_palindrome

input_string=str(input("Enter a String: "))
result=long_palindrome(input_string)
print("Longest palindromic substring:",result)




#6.Merge Intervals

def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged=[intervals[0]]
    for interval in intervals[1:]:
        if interval[0]<=merged[-1][1]:
            merged[-1]=[merged[-1][0],max(merged[-1][1], interval[1])]
        else:
            merged.append(interval)
    return merged

intervals=[[1,3],[2,6],[8,10],[15,18]]
merged_intervals=merge_intervals(intervals)
print("Merged Intervals:",merged_intervals)



#7.Maximum Subarray

def maximum_sub(nums):
    if not nums:
        return 0
    
    max_sum=nums[0]
    current_sum=nums[0]

    for num in nums[1:]:
        current_sum=max(num,current_sum+num)
        max_sum=max(max_sum,current_sum)
    return max_sum

nums=[-2,1,-3,4,-1,2,1,-5,4]
max_sub=maximum_sub(nums)
print("Maximum subarray sum:",max_sub)



#8.Reverse Linked List

class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_linked_list(head):
    prev = None
    current = head

    while current:
        next_node = current.next  
        current.next = prev  
        prev = current  
        current = next_node  

    return prev  


def print_linked_list(head):
    current = head
    while current:
        print(current.value, end=" -> ")
        current = current.next
    print("None")

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)

reversed_head = reverse_linked_list(head)

print("Reversed Linked List:")
print_linked_list(reversed_head)



#9.Minimum Edit Distance

def min_edit_distance(word1, word2):
    
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]

    for i in range(len(word1) + 1):
        dp[i][0] = i
    for j in range(len(word2) + 1):
        dp[0][j] = j

    
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] 
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  
                                   dp[i][j - 1], 
                                   dp[i - 1][j - 1]) 

    return dp[len(word1)][len(word2)]


word1 = "kitten"
word2 = "sitting"
distance = min_edit_distance(word1, word2)
print("Minimum Edit Distance:", distance)



#10.Boggle Game

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

class BoggleSolver:
    def __init__(self, board, dictionary):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.trie = Trie()
        for word in dictionary:
            self.trie.insert(word)
        self.words = set()

    def find_words(self):
        for i in range(self.rows):
            for j in range(self.cols):
                visited = [[False] * self.cols for _ in range(self.rows)]
                self.dfs(i, j, "", self.trie.root, visited)
        return list(self.words)

    def dfs(self, i, j, current_word, trie_node, visited):
        if i < 0 or i >= self.rows or j < 0 or j >= self.cols or visited[i][j]:
            return

        char = self.board[i][j]
        if char not in trie_node.children:
            return

        current_word += char
        trie_node = trie_node.children[char]

        if trie_node.is_end_of_word:
            self.words.add(current_word)

        visited[i][j] = True

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_i, new_j = i + dx, j + dy
                if 0 <= new_i < self.rows and 0 <= new_j < self.cols and not visited[new_i][new_j]:
                    self.dfs(new_i, new_j, current_word, trie_node, visited)

        visited[i][j] = False

board = [
    ['o', 'a', 't'],
    ['e', 't', 'a'],
    ['t', 'r', 'e']
]
dictionary = ["eat", "oat", "tea", "rate"]

boggle_solver = BoggleSolver(board, dictionary)
words_found = boggle_solver.find_words()
print("Words found:", words_found)