
#ifndef BC_CONTEXT_HOSTSTREAM_H_
#define BC_CONTEXT_HOSTSTREAM_H_

#include <thread>
#include <queue>
#include <mutex>
#include <memory>
#include <atomic>
#include <condition_variable>

namespace BC {
namespace streams {

class HostEvent {

	struct contents {
		std::atomic_bool recorded{false};
		std::condition_variable cv;
		std::mutex m_mutex;
	};

	struct waiting_functor {
		std::shared_ptr<contents> m_contents;
		void operator () () const {
			std::unique_lock<std::mutex> locker(m_contents.get()->m_mutex);

			if (!m_contents.get()->recorded.load())
				m_contents.get()->cv.wait(locker, [&](){ return m_contents.get()->recorded.load(); });
		}
	};

	struct recording_functor {
		std::shared_ptr<contents> m_contents;

		void operator () () const {
			m_contents.get()->recorded.store(true);
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
		virtual void operator () () const override final { f(); }
		virtual void run() 			const override final { f(); }
	};

	mutable std::mutex m_queue_lock;
	std::queue<std::unique_ptr<Job>> m_queue;

private:

	void execute_queue() {
		m_queue_lock.lock();		//lock while checking if empty
		while (!m_queue.empty()){
			std::unique_ptr<Job> curr_job = std::move(m_queue.front()); //move the current job
			m_queue.pop();												//remove from queue
			m_queue_lock.unlock();	//unlock  while executing
			curr_job.get()->run();	//this allows more jobs to be added while executing
			m_queue_lock.lock();	//reacquire mutex (while we check if its empty)
		}
		m_queue_lock.unlock();		//unlock
	}

public:
	template<class function>
	void push(function functor) {
		m_queue_lock.lock();
		if (m_queue.empty()) {
			m_queue_lock.unlock();
				BC_omp_async__(
					functor();
					execute_queue();
				)
		} else {
			m_queue.push(std::unique_ptr<Job>(new JobInstance<function>(functor)));
			m_queue_lock.unlock();
		}
	}

	bool empty() const {
		return m_queue.empty();
	}

	bool active() const {
		bool is_active;
		m_queue_lock.lock();
		is_active = m_queue.empty();
		m_queue_lock.unlock();
		return is_active;
	}
}; //End of Queue object


}
}

#endif
