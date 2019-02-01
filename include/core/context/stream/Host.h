
#ifndef HOST_H_
#define HOST_H_

#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <condition_variable>

namespace BC {
namespace context {

class HostQueue {

	struct Job {

		virtual void operator () () const = 0;
		virtual void run () const = 0;

		virtual ~Job() {};
	};

	template<class function>
	struct JobInstance : Job {

		function f;

		JobInstance(function f) : f(f) {}

		void operator () () const override {
			f();
		}

		virtual void run() const override  {
			f();
		}
	};

	bool m_final_terminate = false;
	std::condition_variable cv;

	std::mutex m_stream_lock;
	std::mutex m_queue_lock;
	std::queue<std::unique_ptr<Job>> m_queue;
	std::unique_ptr<std::thread> m_stream;

	void run() {

		while (!m_final_terminate) {
			std::unique_lock<std::mutex> unq_stream_lock(m_stream_lock);
			cv.wait(unq_stream_lock, [&](){ return !m_queue.empty();});

			while (!m_queue.empty()) {
				m_queue_lock.lock();
				std::unique_ptr<Job> curr_job_ = std::move(m_queue.front());
				m_queue.pop();
				m_queue_lock.unlock();

				curr_job_.get()->run();
			}
		}
	}



public:

	void init() {
		this->m_final_terminate = false;
		m_stream = std::unique_ptr<std::thread>(new std::thread(&HostQueue::run, this));
	}


	template<class function>
	void push(function functor) {
		m_queue_lock.lock();
		m_queue.push(job_handle_t(new JobInstance<function>(functor)));
		m_queue_lock.unlock();
		cv.notify_one();
	}

	bool empty() const {
		return m_queue.empty();
	}

	bool active() const {
		return m_stream && !m_final_terminate;
	}

	void terminate() {
		this->m_final_terminate = true;
	}

	void synchronize() {
		if (this->active()) {
			terminate();
			m_stream.get()->join();
			init();
		}
	}

	~HostQueue() {
		terminate();
		m_stream.get()->join();
	}

}; //End of Queue object


}
}

#endif
