# ENPM_661_Project_2
 Project 2 implementing Dijkstra and BFS Search for a point robot navigating obstacles

Chris Collins, 

GitHub: https://github.com/Cgcollins91/Project_2

I created my project 2 code in VS Code using code blocks. 

The first block imports the necessary libraries, and all the functions I wrote for Project 2

The second block is used to define 1 of 3 test cases and the path to save the output videos, then run that case
- IMPORTANT: users should edit the save_folder_path by creating a list of the folder structure 
   after the user's base directory to point the program where to save the videos
   i.e. in my case, I had it save to "C:\Users\Chris\Dropbox\UMD\ENPM_671 - Path Planning for Robots"
   so the input provided is ["Dropbox", "UMD", "ENPM_671 - Path Planning for Robots"]

Test Cases:
  - case = 1 : Runs a pre-defined start/goal-state from the top-left corner of the map to the bottom right corner
  - case = 2 : Prompt user to enter start/goal-state coordinates in terminal
  - case = 3 : Generate Random Valid Start/Goal-States

The code is executed as follows:
    Step 1: Create Map with Obstacles, Map with Obstacles + 2mm clearence and Cost Matrix
    Step 2: Get Start/ Goal State, either from user or generate random valid start/ goal state
    Step 3: Run Dijkstra's Algorithm or BFS and plot HeatMap of Cost Matrix
    Step 4: Create Output Video of solution path and explored path


For Djikstra's Algorithm:

    Data Structure Selection:
        We use a priority queue (heapq) as our Open List to add nodes to visit as we search
        Priority queues are type of queue where each element is associated with priority and element with the highest priority is served
        before an element with lower priority -- Priority queues are implemented as binary heaps that have one of the following properties:
         - Min-Heap Property: For every node in the tree, the key is Less than or equal to the keys of its children
         - Max-Heap Property: For every node in the tree, the key is Greater than or equal to the keys of its children
        Priority queues work well with algorithms like Dijkstra's where we need to visit nodes in order of their cost

        We use a set to track visited nodes in our Closed List, cost_to_come to track the cost to each node
        We use a dictionary to map child nodes to parent nodes to back-track our path from goal to start state
        We use a list for explored_path to track all nodes visited in order to visualize the path taken by the algorithm

    Algorithm:
        1. Initialize priority queue (pq) with start state and cost_to_come dictionary with start state cost as 0
        2. Initialize parent dictionary with start state as None
        3. Initialize cost_matrix with start state cost as 0
        4. While pq is not empty, pop the node with the lowest cost from pq
        5. If goal state is reached, generate path from start to goal and break the loop
        6. Generate all possible moves from current state
        7. For each move, check if it is valid and not an obstacle
        8. If move is valid and not an obstacle, calculate new cost to reach the node
        9. If new cost is lower than previous cost, update cost_to_come, parent, cost_matrix, and add node to pq
        10. If node is reached again with a lower cost, we skip it
        11. If no solution is found, print "No Solution Found"
        12. Return solution path, cost_to_come, parent, cost_matrix, and explored_path

    Parameters:
        start_state: Initial state of point robot as tuple of (x, y) coordinates
        goal_state:  Goal state of point robot as tuple of (x, y) coordinates
        map_data:    Map with obstacles
        cost_matrix: Cost matrix with obstacles as -1 and free space as infinity
        obstacles:   Set of obstacle coordinates

    Returns:     
        solution_path: List of states from the start state to goal state
        cost_to_come:  Dictionary of cost to reach each node
        parent:        Dictionary mapping child states to parent states
        cost_matrix:   Cost matrix with updated costs to reach each node
        explored_path: List of all nodes expanded by the algorithm in search


In dijkstra_chris_collins.py I added a 3rd and 4th cell, where the 3rd cell runs a A_star using octile distance as our heuristic funciton 
and the 4th cell runs some test edge cases

Note for the BFS implementation the Cost Matrix is not a true cost, it just shows the order in which the nodes were visited sequentially 
