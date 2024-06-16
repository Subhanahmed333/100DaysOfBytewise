# #1. QuickSort Algorithm

def quicksort(arr):
    if len(arr)<=1:
        return arr
    else:
        pivot=arr[-1]
        larger=[x for x in arr[:-1] if x>pivot]
        smaller=[x for x in arr[:-1] if x<=pivot]
        return quicksort(smaller)+[pivot]+quicksort(larger)
    
arrinput=[3,6,8,10,1,2,1]
sorted_quick=quicksort(arrinput)
print(sorted_quick)


# #2.Knapsack Problem

maximum_capacity=7
no_of_items=4
weight=[1,3,4,5]
values=[1,4,5,7]
K=[[0 for w in range(maximum_capacity+1)]for i in range(no_of_items+1)]
for i in range(1,no_of_items+1):
    for j in range(1,maximum_capacity+1):
        if weight[i-1]<=j:
             K[i][j] = max(values[i-1] + K[i-1][j-weight[i-1]], K[i-1][j])
        else:
            K[i][j] = K[i-1][j]
        
items_included = []
i, j = no_of_items, maximum_capacity
while i > 0 and j > 0:
    if K[i][j] != K[i-1][j]:
        items_included.append(i-1)
        j -= weight[i-1]
    i -= 1

print("Maximum value:", K[no_of_items][maximum_capacity])



#3.Graph Traversal (BFS and DFS)
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self, start_node):
        visited = set()
        queue = deque([start_node])
        result = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                result.append(node)
                visited.add(node)
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return result

    def dfs(self, start_node):
        visited = set()
        result = []

        def dfs_recursive(node):
            visited.add(node)
            result.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)

        dfs_recursive(start_node)
        return result
    
if __name__ == "__main__":
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)
    graph.add_edge(2, 3)
    graph.add_edge(3, 3)

    print("BFS starting from node 2:", graph.bfs(2))
    print("DFS starting from node 2:", graph.dfs(2))



#4.Dijkstra's Algorithm

import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    
    priority_queue = [(0, start)]
    
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

if __name__ == "__main__":
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 5},
        'C': {'D': 1},
        'D': {}
    }
    start_node = 'A'
    shortest_distances = dijkstra(graph, start_node)
    print("Shortest distances from node", start_node, ": ", shortest_distances)



#5.Longest Common Subsequence (LCS)
def longest_common_subsequence(s1, s2):
    m = len(s1)
    n = len(s2)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    index = lcs[m][n]
    lcs_string = [""] * (index + 1)
    lcs_string[index] = ""
    
    i = m
    j = n
    while i > 0 and j > 0: 
        if s1[i - 1] == s2[j - 1]:
            lcs_string[index - 1] = s1[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif lcs[i - 1][j] > lcs[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(lcs_string)

if __name__ == "__main__":
    s1 = "AGGTAB"
    s2 = "GXTXAYB"
    result = longest_common_subsequence(s1, s2)
    print("Longest Common Subsequence:", result)
