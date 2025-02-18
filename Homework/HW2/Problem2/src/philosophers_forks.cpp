// Program for solving the dining philosophers problem using pthreads
// Each philosopher is a thread, and the forks are the synchronizations needed between them.
// This version provides an updating text-based GUI without ncurses.
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
        update_status(id, THINKING);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, 10);
        sleep(distr(gen));
    }
    
    // Update the status of the philosopher to eating for a random amount of time
    void eat(int id) {
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is eating...");
        }
        update_status(id, EATING);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, 10);
        sleep(distr(gen));
    }

    // Pick up the forks and test if the philosopher can eat
    void pickup_forks(int id) {
        pthread_mutex_lock(&mutex);
        states[id] = HUNGRY;
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is picking up the forks.");
        }
        update_display();
        test(id);
        pthread_mutex_unlock(&mutex);
        pthread_mutex_lock(&forks[id]);  // Wait until allowed to eat
    }

    // Put down the forks and test if neighbors can eat
    void putdown_forks(int id) {
        pthread_mutex_lock(&mutex);
        states[id] = THINKING;
        if (!USE_FANCY_DISPLAY) {
            print_safe("Philosopher " + std::to_string(id) + " is putting down the forks.");
        }
        test((id + num_philosophers - 1) % num_philosophers); // Test left neighbor
        test((id + 1) % num_philosophers); // Test right neighbor
        pthread_mutex_unlock(&mutex);
        update_display();
    }

    // Test if the philosopher can eat
    void test(int id) {
        if (states[id] == HUNGRY &&
            states[(id + num_philosophers - 1) % num_philosophers] != EATING &&
            states[(id + 1) % num_philosophers] != EATING) {
            states[id] = EATING;
            pthread_mutex_unlock(&forks[id]);
            update_display();
        }
    }

    // Update the status of the philosopher to the given state
    void update_status(int id, int state) {
        pthread_mutex_lock(&mutex);
        states[id] = state;
        update_display();
        pthread_mutex_unlock(&mutex);
    }

    // Updates the display with the current state of the philosophers in a fancy, colored way
    void update_display() {
        if (!USE_FANCY_DISPLAY) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Small delay to reduce flickering
        std::cout << "\033[H\033[J"; // Clears the terminal screen

        std::cout << "Dining Philosophers Simulation\n";
        std::cout << "------------------------------\n";

        for (int i = 0; i < num_philosophers; ++i) {
            std::cout << "Philosopher " << i << ": ";
            switch (states[i]) {
                case THINKING:
                    std::cout << "\033[33mTHINKING\033[0m"; // Yellow
                    break;
                case HUNGRY:
                    std::cout << "\033[31mHUNGRY  \033[0m"; // Red
                    break;
                case EATING:
                    std::cout << "\033[32mEATING  \033[0m"; // Green
                    break;
            }
            std::cout << "\n";
        }
        std::cout.flush();
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
