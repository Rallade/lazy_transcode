"""
Comprehensive test runner for enhanced transcoding functionality.

This test suite covers all the enhancements made to address logging
and stream preservation concerns in the transcoding engine.
"""

import unittest
import sys
from pathlib import Path

# Add the lazy_transcode package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test modules
from tests.test_enhanced_transcoding import (
    TestStreamAnalysisLogging,
    TestEnhancedVBRTranscoding, 
    TestProgressMonitoring as TestProgressMonitoringBasic,
    TestStreamPreservationIntegration,
    TestProgressFileHandling
)

from tests.test_stream_preservation import (
    TestEncoderConfigBuilderStreamPreservation,
    TestVBROptimizerIntegration,
    TestStreamPreservationCommand,
    TestErrorHandlingInStreamPreservation
)

from tests.test_progress_monitoring import (
    TestProgressMonitoring,
    TestProgressDataParsing
)


def create_test_suite():
    """Create comprehensive test suite for enhanced transcoding functionality."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Enhanced transcoding tests
    suite.addTests(loader.loadTestsFromTestCase(TestStreamAnalysisLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedVBRTranscoding))
    suite.addTests(loader.loadTestsFromTestCase(TestProgressMonitoringBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamPreservationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestProgressFileHandling))
    
    # Stream preservation tests
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderConfigBuilderStreamPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestVBROptimizerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamPreservationCommand))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingInStreamPreservation))
    
    # Progress monitoring tests
    suite.addTests(loader.loadTestsFromTestCase(TestProgressMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestProgressDataParsing))
    
    return suite


def run_enhanced_transcoding_tests(verbosity=2):
    """Run all enhanced transcoding tests with detailed output."""
    print("=" * 80)
    print("RUNNING ENHANCED TRANSCODING FUNCTIONALITY TESTS")
    print("=" * 80)
    print()
    print("Testing the following enhancements:")
    print("✓ Comprehensive transcoding logging")
    print("✓ Input/output stream analysis")
    print("✓ Audio, subtitle, and chapter preservation") 
    print("✓ Enhanced progress monitoring")
    print("✓ Error handling and reporting")
    print("✓ Integration with EncoderConfigBuilder")
    print()
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    
    print("Running tests...")
    print("-" * 40)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests run: {total_tests}")
    print(f"✓ Passed: {passed}")
    if failures > 0:
        print(f"✗ Failed: {failures}")
    if errors > 0:
        print(f"✗ Errors: {errors}")
    if skipped > 0:
        print(f"- Skipped: {skipped}")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFAILURE DETAILS:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"FAIL: {test}")
            print(traceback)
            print()
    
    if result.errors:
        print("\nERROR DETAILS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(traceback)
            print()
    
    # Overall result
    if failures == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED! Enhanced transcoding functionality is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Review the details above.")
        return False


def run_individual_test_categories():
    """Run tests by category for detailed analysis."""
    categories = [
        ("Stream Analysis & Logging", [
            TestStreamAnalysisLogging
        ]),
        ("VBR Transcoding Enhancements", [
            TestEnhancedVBRTranscoding,
            TestStreamPreservationIntegration,
            TestProgressFileHandling
        ]),
        ("Stream Preservation", [
            TestEncoderConfigBuilderStreamPreservation,
            TestVBROptimizerIntegration,
            TestStreamPreservationCommand,
            TestErrorHandlingInStreamPreservation
        ]),
        ("Progress Monitoring", [
            TestProgressMonitoring,
            TestProgressDataParsing
        ])
    ]
    
    overall_success = True
    
    for category_name, test_classes in categories:
        print(f"\n{'=' * 60}")
        print(f"CATEGORY: {category_name}")
        print(f"{'=' * 60}")
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        runner = unittest.TextTestRunner(verbosity=1, buffer=True)
        result = runner.run(suite)
        
        category_success = len(result.failures) == 0 and len(result.errors) == 0
        overall_success = overall_success and category_success
        
        status = "✅ PASSED" if category_success else "❌ FAILED"
        print(f"{category_name}: {status}")
    
    return overall_success


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced transcoding tests')
    parser.add_argument('--category', action='store_true', 
                      help='Run tests by category')
    parser.add_argument('--verbose', '-v', action='count', default=2,
                      help='Increase verbosity level')
    
    args = parser.parse_args()
    
    if args.category:
        success = run_individual_test_categories()
    else:
        success = run_enhanced_transcoding_tests(verbosity=args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
