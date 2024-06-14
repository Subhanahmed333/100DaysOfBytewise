#1. QuickSort Algorithm

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


#2.Knapsack Problem

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

# Print results
print("Maximum value:", K[no_of_items][maximum_capacity])
