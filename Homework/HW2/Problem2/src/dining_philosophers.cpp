// Program for solving the Dining Philosophers problem using pthreads
// Each philosopher is a thread, and the forks are the synchronizations (mutexes) needed between them.
// Philosophers think for a random amount of time and eat for a random amount of time. They are 'HUNGRY' if waiting for forks.
// Author: Zander Ingare

#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <random>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

#define THINKING 0
#define HUNGRY 1
#define EATING 2
#define USE_FANCY_DISPLAY true

// Global mutex for synchronizing cout since the start of the program creates janky output
pthread_mutex_t cout_mutex = PTHREAD_MUTEX_INITIALIZER;

class DiningPhilosophers {
private:
    int num_philosophers;
    std::vector<pthread_t> philosophers;
    std::vector<pthread_mutex_t> forks;
    std::vector<int> states;
    std::vector<int> forks_taken;
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

    // Update the status of the philosopher to thinking for a random amount of time
    void think(int id) {
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is thinking...");
        }
        update_philosopher_status(id, THINKING);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(2, 5);
        sleep(distr(gen));
    }

    // Pick up the forks and check if the philosopher can eat
    void pickup_forks(int id) {
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is picking up the forks.");
        }
        update_philosopher_status(id, HUNGRY);

        int left_fork = (id + num_philosophers - 1) % num_philosophers;
        int right_fork = id;

        // Lock left fork first and update its status immediately
        pthread_mutex_lock(&forks[left_fork]);
        pthread_mutex_lock(&mutex);
        forks_taken[left_fork] = id;  // Mark fork as taken by this philosopher
        pthread_mutex_unlock(&mutex);
        update_display();  // Reflect fork lock immediately

        // Now attempt to lock the right fork and update its status
        pthread_mutex_lock(&forks[right_fork]);
        pthread_mutex_lock(&mutex);
        forks_taken[right_fork] = id;  // Mark fork as taken
        pthread_mutex_unlock(&mutex);
        update_display();  // Reflect second fork lock immediately
    }

    // Update the status of the philosopher to eating for a random amount of time
    void eat(int id) {
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is eating...");
        }
        update_philosopher_status(id, EATING);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(5, 10);
        sleep(distr(gen));
    }

    // Put down the forks and relinquish their locks
    void putdown_forks(int id) {
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is putting down the forks.");
        }
        update_philosopher_status(id, THINKING);

        int left_fork = (id + num_philosophers - 1) % num_philosophers;
        int right_fork = id;

        // Unlock both forks and mark them as available immediately
        pthread_mutex_lock(&mutex);
        forks_taken[left_fork] = -1;
        forks_taken[right_fork] = -1;
        pthread_mutex_unlock(&mutex);

        pthread_mutex_unlock(&forks[left_fork]);
        pthread_mutex_unlock(&forks[right_fork]);
        
        update_display();  // Reflect that the forks are now available
    }

    // Update the status of the philosopher to the given state
    void update_philosopher_status(int id, int state) {
        pthread_mutex_lock(&mutex);
        states[id] = state;
        update_display();
        pthread_mutex_unlock(&mutex);
    }

    // Updates the display with the current state of the philosophers and forks
    void update_display() {
        if (!USE_FANCY_DISPLAY) {
            return;
        }
        pthread_mutex_lock(&cout_mutex);
        std::cout << "\033[H\033[J"; // Clears the terminal screen

        std::cout << "Dining Philosophers Simulation\n";
        std::cout << "------------------------------\n";

        for (int i = 0; i < num_philosophers; ++i) {
            std::cout << "Philosopher " << i << ": ";
            switch (states[i]) {
                case THINKING:
                    std::cout << "\033[33m THINKING \033[0m"; // Yellow
                    break;
                case HUNGRY:
                    std::cout << "\033[31m HUNGRY \033[0m"; // Red
                    break;
                case EATING:
                    std::cout << "\033[32m EATING \033[0m"; // Green
                    break;
            }
            std::cout << "\n";

            // Display fork ownership based on locks taken (THIS TOOK ME FOREVER TO DEBUG)
            std::cout << "  └── Fork " << i << ": ";
            if (forks_taken[i] != -1) {
                std::cout << "\033[36m Taken by Philosopher \033[0m" << forks_taken[i]; // Cyan
            } else {
                std::cout << "\033[37m Available \033[0m"; // Gray
            }
            std::cout << "\n";
        }

        std::cout.flush();
        pthread_mutex_unlock(&cout_mutex);
    }

    // Thread-safe cout output function
    void print_safe(const std::string& message) {
        if (!USE_FANCY_DISPLAY) {
            return;
        }
        pthread_mutex_lock(&cout_mutex);
        std::cout << message << std::endl;
        pthread_mutex_unlock(&cout_mutex);
    }

public:
    DiningPhilosophers(int num) : num_philosophers(num) {
        philosophers.resize(num);
        forks.resize(num);
        states.resize(num, THINKING);
        forks_taken.resize(num, -1);
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

    // Create the philosopher threads and start the dining process. Terminates when all threads are done
    void start_dining() {
        update_display();
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

    int num_philosophers = std::atoi(argv[1]);
    if (num_philosophers <= 1 || num_philosophers % 2 == 0) {
        std::cerr << "Number of philosophers must be an odd number greater than 1.\n";
        return 1;
    }

    DiningPhilosophers dp(num_philosophers);
    dp.start_dining();

    return 0;
}
