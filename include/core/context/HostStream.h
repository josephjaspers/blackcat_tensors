
#ifndef HOST_H_
#define HOST_H_

#include <thread>
#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>

namespace BC {
namespace context {

class HostEvent {

	struct contents {
		bool recorded = false;
		std::condition_variable cv;
		std::mutex m_mutex;
	};

	struct waiting_functor {
		std::shared_ptr<contents> m_contents;
		void operator () () const {
			std::unique_lock<std::mutex> locker(m_contents.get()->m_mutex);

			if (!m_contents.get()->recorded)
				m_contents.get()->cv.wait(locker, [&](){ return m_contents.get()->recorded; });
		}
	};

	struct recording_functor {
		std::shared_ptr<contents> m_contents;

		void operator () () const {
			m_contents.get()->recorded = true;
			m_contents.get()->cv.notify_all();
		}
	};

	std::shared_ptr<contents> m_contents = std::shared_ptr<contents>(new contents());

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


	using scoped_lock = std::unique_lock<std::mutex>;
	bool m_final_terminate = false;
	std::condition_variable cv;

	std::mutex m_queue_lock;
	std::mutex m_stream_lock;
	std::queue<std::unique_ptr<Job>> m_queue;
	std::unique_ptr<std::thread> m_stream;

	void run() {

		while (!m_final_terminate) {
			{
				scoped_lock lock(m_stream_lock);
				cv.wait(lock, [&](){ return !m_queue.empty() || m_final_terminate;});
			}
			while (!m_queue.empty()) {
				std::unique_ptr<Job> curr_job_;
				{
					scoped_lock lock(m_queue_lock);
					curr_job_ = std::move(m_queue.front());
					m_queue.pop();
				}
				curr_job_.get()->run();
			}
		}
		//finally finish all the jobs
		m_queue_lock.lock();
		std::queue<std::unique_ptr<Job>> queue_ = std::move(m_queue);
		m_queue_lock.unlock();

		while (!queue_.empty()) {
			queue_.front().get()->run();
			queue_.pop();
		}
	}



public:

	void init() {
		scoped_lock lock(m_queue_lock);
		if (m_stream.get() == nullptr) {
			this->m_final_terminate = false;
			m_stream = std::unique_ptr<std::thread>(new std::thread(&HostStream::run, this));
		}
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

	~HostStream() {
		terminate();
	}

}; //End of Queue object


}
}

#endif
