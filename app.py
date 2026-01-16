import argparse


def main():
    """Entry point for GDA Application"""
    parser = argparse.ArgumentParser(
        description="GDA System - Global Description Acquisition"
    )
    parser.add_argument(
        '--seg-checkpoint', 
        type=str,
        default=None,
        help='Segmentation decoder checkpoint (COCO-Stuff 171 classes)'
    )
    parser.add_argument(
        '--adaptor-checkpoint', 
        type=str,
        default=None,
        help='Adaptor checkpoint (optional)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup for --help
    from src.app import GDAApplication
    
    app = GDAApplication(
        seg_checkpoint=args.seg_checkpoint,
        adaptor_checkpoint=args.adaptor_checkpoint,
        debug=args.debug
    )
    
    app.run()


if __name__ == "__main__":
    main()