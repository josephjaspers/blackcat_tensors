
#ifndef HOST_H_
#define HOST_H_

#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <condition_variable>

namespace BC {
namespace context {

class HostEvent {

	struct contents {
		bool recorded = false;
		std::condition_variable cv;
		std::mutex m;
	};
	struct waiting_functor {
		std::shared_ptr<contents> m_contents;
		void operator () () const {
			if (!m_contents.get()->recorded) {
				std::unique_lock<std::mutex> lock = std::unique_lock<std::mutex>(m_contents.get()->m);

				//check again in case recorded was written to inbetween accessing the lock
				if (!m_contents.get()->recorded) {
					m_contents.get()->cv.wait(lock, [&](){ return m_contents.get()->recorded; });
				}
			}
		}
	};
	struct recording_functor {
		std::shared_ptr<contents> m_contents;

		void operator () () const {
			m_contents.get()->recorded = true;
			m_contents.get()->cv.notify_all();
		}
	};
	std::shared_ptr<contents> m_contents = std::shared_ptr<contents>(new contents);
public:
	recording_functor get_recorder() {
		return {m_contents};
	}
	waiting_functor get_waiter() {
		return {m_contents};
	}
};

class HostStream {

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
			cv.wait(unq_stream_lock, [&](){ return !m_queue.empty() || m_final_terminate;});

			while (!m_queue.empty()) {
				m_queue_lock.lock();
				std::unique_ptr<Job> curr_job_ = std::move(m_queue.front());
				m_queue.pop();
				m_queue_lock.unlock();

				curr_job_.get()->run();
			}
		}
		//finally finish all the jobs
		while (!m_queue.empty()) {
			std::unique_ptr<Job> curr_job_ = std::move(m_queue.front());
			m_queue.pop();
			curr_job_.get()->run();
		}
	}



public:

	void init() {
		this->m_final_terminate = false;
		m_stream = std::unique_ptr<std::thread>(new std::thread(&HostStream::run, this));
	}


	template<class function>
	void push(function functor) {
		m_queue_lock.lock();
		m_queue.push(std::unique_ptr<Job>(new JobInstance<function>(functor)));
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
			if (m_stream.get()) {

			m_final_terminate = true;
			cv.notify_one();

			if (m_stream.get() && m_stream.get()->joinable()) {
				m_stream.get()->join();
			}
		}
	}

	void synchronize() {
		if (this->active()) {
			terminate();
			init();
		}
	}

	~HostStream() {
		terminate();
	}

}; //End of Queue object


}
}

#endif
