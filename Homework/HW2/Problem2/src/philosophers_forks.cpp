// Program for finding the solution to the dining philosophers problem using pthreads
// The solution should be for an unbounded odd number of philosophers, with each philosopher implemented as a thread, and the forks are the synchronizations needed between them.
// Author: Zander Ingare

#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <random>
#include <cstdlib>
#include <vector>

#define THINKING 0
#define HUNGRY 1
#define EATING 2

class DiningPhilosophers {
private:
    int num_philosophers;
    std::vector<pthread_t> philosophers;
    std::vector<pthread_mutex_t> forks;
    std::vector<int> states;
    pthread_mutex_t mutex;

    static void* philosopher_thread(void* arg) {
        auto* self = static_cast<DiningPhilosophers*>(arg);
        int id = pthread_self() % self->num_philosophers;
        while (true) {
            self->think(id);
            self->pickup_forks(id);
            self->eat(id);
            self->putdown_forks(id);
        }
        return nullptr;
    }

    void think(int id) {
        std::cout << "Philosopher " << id << " is thinking..." << std::endl;
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_int_distribution<> distr(1, 20);
        sleep(distr(gen));
    }
    
    void eat(int id) {
        std::cout << "Philosopher " << id << " is eating..." << std::endl;
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_int_distribution<> distr(1, 10);
        sleep(distr(gen));
    }

    void pickup_forks(int id) {
        pthread_mutex_lock(&mutex);
        states[id] = HUNGRY;
        std::cout << "Philosopher " << id << " is hungry and trying to pick up forks.\n";
        test(id);
        pthread_mutex_unlock(&mutex);
        pthread_mutex_lock(&forks[id]);  // Wait until allowed to eat
    }

    void putdown_forks(int id) {
        pthread_mutex_lock(&mutex);
        states[id] = THINKING;
        std::cout << "Philosopher " << id << " is putting down forks.\n";
        test((id + num_philosophers - 1) % num_philosophers); // Test left neighbor
        test((id + 1) % num_philosophers); // Test right neighbor
        pthread_mutex_unlock(&mutex);
    }

    void test(int id) {
        if (states[id] == HUNGRY &&
            states[(id + num_philosophers - 1) % num_philosophers] != EATING &&
            states[(id + 1) % num_philosophers] != EATING) {
            states[id] = EATING;
            pthread_mutex_unlock(&forks[id]);
        }
    }

public:
    DiningPhilosophers(int num) : num_philosophers(num) {
        philosophers.resize(num);
        forks.resize(num);
        states.resize(num, THINKING);
        pthread_mutex_init(&mutex, nullptr);
        for (auto& fork : forks) {
            pthread_mutex_init(&fork, nullptr);
        }
    }

    ~DiningPhilosophers() {
        for (auto& fork : forks) {
            pthread_mutex_destroy(&fork);
        }
        pthread_mutex_destroy(&mutex);
    }

    void start_dining() {
        for (int i = 0; i < num_philosophers; ++i) {
            pthread_create(&philosophers[i], nullptr, philosopher_thread, this);
        }
        for (auto& philosopher : philosophers) {
            pthread_join(philosopher, nullptr);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_philosophers>\n";
        return 1;
    }
    int num_philosophers = atoi(argv[1]);
    if (num_philosophers <= 1 || num_philosophers % 2 == 0) {
        std::cerr << "Number of philosophers must be an odd number greater than 1.\n";
        return 1;
    }

    DiningPhilosophers dp(num_philosophers);
    dp.start_dining();

    return 0;
}
