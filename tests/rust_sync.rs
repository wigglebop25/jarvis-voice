use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

#[test]
fn test_internal_signaling() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair_clone = pair.clone();

    // Simulate worker thread
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        let (lock, cvar) = &*pair_clone;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one();
    });

    // Wait for signal
    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cvar.wait(started).unwrap();
    }
    assert!(*started);
}

#[test]
fn test_signaling_timeout() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair_clone = pair.clone();

    // Simulate slow worker thread
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        let (lock, cvar) = &*pair_clone;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one();
    });

    // Wait with timeout
    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    let (guard, res) = cvar
        .wait_timeout(started, Duration::from_millis(100))
        .unwrap();
    started = guard;

    assert!(res.timed_out());
    assert!(!*started);
}

#[test]
fn test_signaling_with_state_reset() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair_clone = pair.clone();

    // Start 1
    thread::spawn({
        let pair = pair_clone.clone();
        move || {
            thread::sleep(Duration::from_millis(50));
            let (lock, cvar) = &*pair;
            let mut completed = lock.lock().unwrap();
            *completed = true;
            cvar.notify_all();
        }
    });

    {
        let (lock, cvar) = &*pair;
        let mut completed = lock.lock().unwrap();
        while !*completed {
            completed = cvar.wait(completed).unwrap();
        }
        assert!(*completed);
        *completed = false; // Reset for next run
    }

    // Start 2
    thread::spawn({
        let pair = pair_clone.clone();
        move || {
            thread::sleep(Duration::from_millis(50));
            let (lock, cvar) = &*pair;
            let mut completed = lock.lock().unwrap();
            *completed = true;
            cvar.notify_all();
        }
    });

    {
        let (lock, cvar) = &*pair;
        let mut completed = lock.lock().unwrap();
        while !*completed {
            completed = cvar.wait(completed).unwrap();
        }
        assert!(*completed);
    }
}
