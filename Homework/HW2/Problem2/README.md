# Problem 2 (30 Points)

In 1965, Edsger W. Dijkstra described the following problem. Five philosophers sit at a round table with bowls of noodles. Forks are placed between each pair of adjacent philosophers. Each philosopher must alternately think or eat. However, a philosopher can only eat noodles when she has both left and right forks. Each fork can be held by only one philosopher, and each fork is picked up sequentially. A philosopher can use the fork only if it is not being used by another philosopher. Eating takes a random amount of time for each philosopher. After she finishes eating, the philosopher needs to put down both forks, so they become available to others. A philosopher can take the fork on her right or the one on her left as they become available, though cannot start eating before getting both forks. Eating is not limited by the remaining amounts of noodles or stomach space; an infinite supply and an infinite demand are assumed. 

Implement a solution for an unbounded odd number of philosophers, where each philosopher is implemented as a thread, and the forks are the synchronizations needed between them. Develop this threaded program in pthreads. The program takes as an input parameter the number of philosophers. The program needs to print out the state of the table (philosophers and forks) – the format is up to you. 

Answer the following questions: you are not required to implement a working solution to the 3 questions below. 

a.) What happens if only 3 forks are placed in the center of the table, but each philosopher still needs to acquire 2 forks to eat?
b.) What happens to your solution if we give one philosopher higher priority over the other philosophers?
c.) What happens to your solution if the philosophers change which fork is acquired first (i.e., the fork on the left or the right) on each pair of requests? 

Provide clear directions on how you tested your pthreads code so that the TA can confirm that your implementation is working. Provide these directions in a README file which instructs how to run through at least 12 iterations of updating the state of the philosophers and forks around the table. 

In your writeup, also discuss who was Edgar Dijkstra, and what is so important about this dining problem, as it relates to the real world. Make sure to discuss the algorithm that bears his name, Dijkstra’s Algorithm. Cite your sources carefully. 

*Written answers to the questions should be included in your homework 2 write-up in pdf format. You should include your C/C++ program and the README file in the zip file submitted.


## Part (a)

If there are only 3 forks in the center of the table instead of being to the left and right of the philosophers, then no matter how many philosophers there are at the table, only one philosopher at most can be eating at a time. This is because each philosopher needs to acquire two forks to eat, and with only 3 forks available, there will always be a fork missing for at least one philosopher. On the other hand, if one fork is taken by a unique philosopher for every fork (assuming the solution allows for single-fork grabbing), then a deadlock situation occurs where the three philosophers that have one fork are waiting for a fork to become available, and none of them can eat, which leads to an infinite loop of waiting.

## Part (b)

Given that my solution forces philosophers to take a random amount of time to both think and eat, if one philosopher is given higher priority over the other philosophers, then it would get preferential treatment to eat in the cases where it is waiting for a fork that is held by a direct neighbor also waiting for a fork (by taking the fork of the waiting neighbor). This means that it's possible for a prioritized philosopher to eat more frequently than the other philosophers. A more egregious example of priority could be that the prioritized philosopher could take a fork from a neighbor who is actively eating, forcing that neighbor to go into thinking. This would make it more likely for the prioritized philosopher to eat even more frequently than the other philosophers compared to if it just took forks from waiting philosophers.

## Part (c)

Given that my solution has philosophers take the left fork first and then the right fork (circular dependency), it avoids most race conditions that could occur if two philosophers were to take a mutex lock on the same fork at the same time. My solution also drastically reduces the chances of deadlock when all philosophers take their left fork at the same time, resulting in no philosopher being able to take their right fork because I randomize the time to think as well as the time to eat (this avoids the higher likelyhood at the start of the program where all philosophers start in the thinking stage). If instead it was random which fork was acquired first, then it becomes more likely for two philosophers to try to acquire the same fork at the same time, which could lead to race conditions on acquiring the fork, leading to unexpected behavior on which philosopher acquires the fork (particularly at the start of the program). Otherwise, there isn't much difference in the solution if the philosophers change which fork is acquired first since the solution still requires a mutex lock (which can never be locked concurrently).

## Who is Edsger Dijkstra?

Edsger W. Dijkstra was a Dutch computer scientist who made significant contributions to the field of computer science. He was born on May 11, 1930, in Rotterdam, Netherlands, and passed away on August 6, 2002, in Nuenen, Netherlands, following a long struggle with cancer. Dijkstra is very well known for his work on the graph-theory problem of finding the shortest path between two nodes in a graph such that the sum of the weights of its constituent edges is minimized, which is now known as Dijkstra's Algorithm. This algorithm is used in many applications today, such as network routing protocols, transportation systems, and computer networks. Dijkstra later developed the first compiler for the ALGOL-60 programming language along with computer scientiest Jaap A. Zonnefeld. He was awarded the Turing Award in 1972 for his fundamental contributions to developing structured programming languages, where an open letter he wrote called "Go To Statement Considered Harmful" (1968) noted that the usage of "go to" instead of statements such as "if then" led to sloppy programming techniques and that a more rigorous approach with modular units using clear single entrance and exit points should be used instead.

With regards to the Dining Philosophers problem (which was proposed be Edsger Dijkstra), the reason why this problem is so important is that it highlights the challenges of concurrent programming and the need for synchronization mechanisms to prevent deadlocks. Since the philosophers need to acquire two forks to eat, with every fork being shared between two philosophers, it easily showcases the issue of race conditions and deadlocks that can occur when multiple threads are trying to access shared resources. Similarly, the synchronization mechanisms used in the solution to the dining philosophers problem showcases the real-life mechanisms needed to prevent deadlocks and ensure that resources are used efficiently.

Sources:
- https://en.wikipedia.org/wiki/Edsger_W._Dijkstra
- https://www.britannica.com/biography/Edsger-Dijkstra

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```

- To run the program, the following command was used within the makefile directory (where <number_of_philosophers> is the odd number of philosophers at the table):
```./build/dining_philosophers <number_of_philosophers>```