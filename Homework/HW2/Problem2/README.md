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


## Part (b)


## Part (c)


## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```