#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct Event {
    std::uint64_t id{};
    std::uint32_t route{};
    std::uint32_t kind{};
    double value{};
    std::string key;
    std::string payload;
};

struct Result {
    std::uint64_t id{};
    double score{};
    std::string label;
};

struct CacheEntry {
    double value{};
    std::uint64_t hits{};
};

struct WorkerStats {
    std::uint64_t count{};
    std::uint64_t checksum{};
    double sum{};
};

struct WorkerReport {
    std::uint64_t count{};
    std::uint64_t checksum{};
    double sum{};
};

struct Scratch {
    std::array<std::uint64_t, 8> words{};
};

class RingQueue {
public:
    static constexpr std::size_t capacity = 512;

    bool push(Event&& event) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [this] { return size_ < capacity || closed_; });
        if (closed_) {
            return false;
        }
        slots_[tail_] = std::move(event);
        tail_ = (tail_ + 1) % capacity;
        ++size_;
        lock.unlock();
        notEmpty_.notify_one();
        return true;
    }

    bool pop(Event& event) {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [this] { return size_ > 0 || closed_; });
        if (size_ == 0 && closed_) {
            return false;
        }
        event = std::move(slots_[head_]);
        head_ = (head_ + 1) % capacity;
        --size_;
        lock.unlock();
        notFull_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::array<Event, capacity> slots_{};
    std::size_t head_{};
    std::size_t tail_{};
    std::size_t size_{};
    bool closed_{};
    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
};

class Pipeline {
public:
    explicit Pipeline(std::size_t workerCount)
        : workerCount_(std::max<std::size_t>(1, workerCount)),
          stats_(workerCount_),
          recentByWorker_(workerCount_) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            routeWeights_.emplace(i, 0.65 + static_cast<double>((i * 17U) % 71U) / 100.0);
            aliases_.emplace_back(i, "route-" + std::to_string(i));
        }

        scoreTable_.resize(1U << 19U);
        for (std::size_t i = 0; i < scoreTable_.size(); ++i) {
            scoreTable_[i] = 0.75 + std::sin(static_cast<double>(i) * 0.0007) * 0.25;
        }

        for (auto& recent : recentByWorker_) {
            recent.reserve(513);
        }

        scorer_ = [](const Event& event, double base) {
            const double kindFactor = 1.0 + static_cast<double>((event.kind * 13U) % 19U) / 100.0;
            return (base + event.value * 0.0031) * kindFactor;
        };
    }

    ~Pipeline() {
        for (auto& item : cache_) {
            delete item.second;
        }
    }

    void run(const std::vector<Event>& events) {
        constexpr std::size_t batchSize = 1024;
        for (std::size_t begin = 0; begin < events.size(); begin += batchSize) {
            const std::size_t end = std::min(events.size(), begin + batchSize);
            processBatch(events, begin, end);
        }
    }

    void printSummary() {
        const auto count = results_.size();
        std::size_t duplicates = 0;
        for (std::size_t i = 0; i < results_.size(); ++i) {
            for (std::size_t j = i + 1; j < results_.size(); ++j) {
                if (results_[i].id == results_[j].id) {
                    ++duplicates;
                }
            }
        }

        std::vector<double> scores;
        scores.reserve(results_.size());
        for (const auto& result : results_) {
            scores.push_back(result.score);
        }
        std::sort(scores.begin(), scores.end());
        const double median = scores.empty() ? 0.0 : scores[scores.size() / 2];

        std::cout << std::fixed << std::setprecision(5)
                  << "events=" << count
                  << " processed=" << processed_.load()
                  << " median=" << median
                  << " duplicates=" << duplicates
                  << " retained=" << history_.size()
                  << " report-sink=" << reportSink_
                  << " metric-sum=" << metricSum_
                  << '\n';
    }

private:
    void processBatch(const std::vector<Event>& input, std::size_t first, std::size_t last) {
        std::vector<Event> staging(input.begin() + static_cast<std::ptrdiff_t>(first),
                                   input.begin() + static_cast<std::ptrdiff_t>(last));
        RingQueue queue;
        std::vector<std::thread> threads;
        threads.reserve(workerCount_);

        for (std::size_t worker = 0; worker < workerCount_; ++worker) {
            threads.emplace_back([this, &queue, worker] {
                Event event;
                while (queue.pop(event)) {
                    processOne(event, worker);
                }
            });
        }

        for (auto& event : staging) {
            queue.push(std::move(event));
        }
        queue.close();

        for (auto& thread : threads) {
            thread.join();
        }

        std::vector<WorkerReport*> reports;
        reports.reserve(workerCount_);
        for (std::size_t worker = 0; worker < workerCount_; ++worker) {
            reports.push_back(new WorkerReport{
                stats_[worker].count,
                stats_[worker].checksum,
                stats_[worker].sum
            });
        }
        for (const auto* report : reports) {
            reportSink_ ^= report->count + report->checksum +
                           static_cast<std::uint64_t>(std::fabs(report->sum));
        }
    }

    const std::string& resolveAlias(std::uint32_t route) const {
        for (const auto& alias : aliases_) {
            if (alias.first == route) {
                return alias.second;
            }
        }
        return aliases_.front().second;
    }

    double getCachedValue(std::uint64_t key, double weight) {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            ++it->second->hits;
            return it->second->value;
        }

        auto* entry = new CacheEntry{weight * 0.91 + 0.07, 1};
        cache_.emplace(key, entry);
        const double result = entry->value;
        if (cache_.size() > 2048) {
            cache_.erase(cache_.begin());
        }
        return result;
    }

    std::string makeLabel(const Event& event) const {
        const auto secondaryHash = std::hash<std::string>{}(event.key);
        return resolveAlias(event.route) + "-" + std::to_string(secondaryHash & 0xFFFFU);
    }

    void processOne(Event event, std::size_t worker) {
        const auto keyHash = std::hash<std::string>{}(event.key);

        double scale = 1.0;
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            scale = configScale_;
        }

        double routeWeight = 1.0;
        const auto routeIt = routeWeights_.find(event.route);
        if (routeIt != routeWeights_.end()) {
            routeWeight = routeIt->second;
        }

        const double cached = getCachedValue(keyHash, routeWeight);
        const auto tableIndex = static_cast<std::size_t>((keyHash ^ event.id) & (scoreTable_.size() - 1));
        const double tableValue = scoreTable_[tableIndex];
        const double nonlinear = std::sqrt(std::pow(std::fabs(event.value) + 0.031, 1.73) + tableValue * 0.013);

        char* raw = new char[event.payload.size() + 1];
        std::memcpy(raw, event.payload.data(), event.payload.size());
        raw[event.payload.size()] = '\0';
        const auto payloadByte = static_cast<unsigned char>(raw[event.id % event.payload.size()]);

        auto scratch = std::make_unique<Scratch>();
        scratch->words[0] = keyHash;
        scratch->words[1] = event.id;
        scratch->words[2] = static_cast<std::uint64_t>(payloadByte);
        scratch->words[3] = static_cast<std::uint64_t>(tableIndex);

        double score = scorer_(event, (nonlinear + routeWeight + cached + scale) * tableValue);
        if (((keyHash ^ (event.id * 0x9E3779B97F4A7C15ULL)) & 1ULL) != 0ULL) {
            score = score * 1.0017 + static_cast<double>(payloadByte) * 0.00001;
        } else {
            score = score * 0.9983 - static_cast<double>(payloadByte) * 0.00001;
        }
        score += static_cast<double>(scratch->words[0] & 0xFFU) * 1e-9;

        auto& recent = recentByWorker_[worker];
        recent.push_back(score);
        if (recent.size() > 512) {
            recent.erase(recent.begin());
        }
        const double recentMean = std::accumulate(recent.begin(), recent.end(), 0.0) /
                                  static_cast<double>(recent.size());
        score += recentMean * 1e-6;

        const std::string label = makeLabel(event);
        std::ostringstream line;
        line << "event=" << event.id << " route=" << event.route
             << " score=" << std::setprecision(5) << score;

        {
            std::lock_guard<std::mutex> lock(metricsMutex_);
            metricSum_ += score;
            ++kindCounts_[event.kind & 15U];
        }
        processed_.fetch_add(1);

        stats_[worker].count += 1;
        stats_[worker].checksum ^= keyHash + event.id + payloadByte;
        stats_[worker].sum += score;

        Result result{event.id, score, label};
        {
            std::lock_guard<std::mutex> lock(resultsMutex_);
            results_.push_back(std::move(result));
            history_.push_back(label);
        }

        {
            std::lock_guard<std::mutex> lock(logMutex_);
            if ((event.id & 255U) == 0U) {
                std::cout << line.str() << std::endl;
            }
        }
    }

    std::size_t workerCount_;
    std::map<std::uint32_t, double> routeWeights_;
    std::vector<std::pair<std::uint32_t, std::string>> aliases_;
    std::vector<double> scoreTable_;
    std::function<double(const Event&, double)> scorer_;

    std::mutex configMutex_;
    double configScale_{1.0};

    std::mutex cacheMutex_;
    std::unordered_map<std::uint64_t, CacheEntry*> cache_;

    std::mutex metricsMutex_;
    double metricSum_{};
    std::array<std::uint64_t, 16> kindCounts_{};
    std::atomic<std::uint64_t> processed_{0};

    std::mutex resultsMutex_;
    std::vector<Result> results_;
    std::vector<std::string> history_;
    std::mutex logMutex_;

    std::vector<WorkerStats> stats_;
    std::vector<std::vector<double>> recentByWorker_;
    std::uint64_t reportSink_{};
};

int main(int argc, char** argv) {
    std::size_t eventCount = 5000;
    std::size_t workers = std::min<std::size_t>(8, std::max(1U, std::thread::hardware_concurrency()));

    if (argc > 1) {
        eventCount = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        workers = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    }
    workers = std::clamp<std::size_t>(workers, 1, 16);

    std::mt19937_64 rng(0xA17C9E31ULL);
    std::uniform_int_distribution<std::uint32_t> routeDist(0, 255);
    std::uniform_int_distribution<std::uint32_t> kindDist(0, 15);
    std::uniform_real_distribution<double> valueDist(-100.0, 100.0);

    std::vector<Event> events;
    events.reserve(eventCount);
    for (std::size_t i = 0; i < eventCount; ++i) {
        Event event;
        event.id = static_cast<std::uint64_t>(i);
        event.route = routeDist(rng);
        event.kind = kindDist(rng);
        event.value = valueDist(rng);
        event.key = "sensor/" + std::to_string(event.route) + "/sample/" + std::to_string(i);
        event.payload.assign(64, static_cast<char>('a' + (event.route % 26)));
        events.push_back(std::move(event));
    }

    Pipeline pipeline(workers);
    const auto start = std::chrono::steady_clock::now();
    pipeline.run(events);
    const auto finish = std::chrono::steady_clock::now();
    pipeline.printSummary();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "elapsed-ms=" << elapsed << " workers=" << workers << '\n';
    return 0;
}
