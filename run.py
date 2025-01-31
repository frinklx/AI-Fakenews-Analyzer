from fakenews_detector.app import app, cleanup
import multiprocessing
import os
import signal
import sys
import logging
import gc
import torch

def handle_sigterm(signum, frame):
    """Handle termination gracefully"""
    logging.info("Shutting down server...")
    cleanup()
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    sys.exit(0)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    # Set environment variables for better process handling
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress warnings
    
    # Clear any existing memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Run the app with a single worker
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,  # Disable debug mode in production
            use_reloader=False,
            threaded=True,
            processes=1
        )
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
        cleanup()
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        sys.exit(0)
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        cleanup()
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        sys.exit(1) 