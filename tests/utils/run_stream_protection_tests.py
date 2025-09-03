"""
Comprehensive test runner for stream preservation bug prevention.

This runner organizes tests by priority and type to ensure critical
stream preservation bugs are caught first.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_suite():
    """Create a comprehensive test suite organized by priority."""
    
    # Define test priorities
    critical_tests = [
        # Tests that would have directly caught the original bug
        'test_command_validation.TestFFmpegCommandValidation.test_command_contains_all_required_stream_preservation_flags',
        'test_stream_preservation_regression.TestStreamPreservationRegression.test_vbr_function_uses_comprehensive_encoder',
        'test_stream_preservation_regression.TestCommandGenerationIntegration.test_end_to_end_command_generation_includes_streams',
    ]
    
    high_priority_tests = [
        # Command structure and syntax validation
        'test_command_validation.TestFFmpegCommandValidation.test_command_structure_is_valid_ffmpeg_syntax',
        'test_command_validation.TestFFmpegCommandValidation.test_no_conflicting_flags_present',
        'test_command_validation.TestFFmpegCommandValidation.test_vbr_optimizer_generates_comprehensive_commands',
        
        # Stream preservation patterns
        'test_command_validation.TestStreamPreservationPatterns.test_comprehensive_stream_mapping_pattern',
        'test_command_validation.TestStreamPreservationPatterns.test_complete_stream_copying_pattern',
        'test_command_validation.TestStreamPreservationPatterns.test_no_stream_exclusion_patterns',
        
        # Module integration
        'test_stream_preservation_regression.TestModuleIntegrationPoints.test_transcoding_engine_imports_correct_vbr_function',
        'test_stream_preservation_regression.TestModuleIntegrationPoints.test_no_duplicate_vbr_functions_exist',
    ]
    
    medium_priority_tests = [
        # Hardware encoder tests
        'test_command_validation.TestHardwareEncoderStreamPreservation',
        
        # Consistency tests
        'test_command_validation.TestCommandGenerationConsistency',
        
        # Real-world scenarios
        'test_stream_preservation_regression.TestRealWorldScenarios',
    ]
    
    low_priority_tests = [
        # Edge cases
        'test_command_validation.TestCommandGenerationEdgeCases',
        
        # Enhanced functionality
        'test_enhanced_transcoding.TestStreamAnalysisLogging',
        'test_enhanced_transcoding.TestEnhancedVBRTranscoding',
        'test_enhanced_transcoding.TestProgressMonitoring',
        
        # General stream preservation
        'test_stream_preservation.TestEncoderConfigBuilderStreamPreservation',
        'test_stream_preservation.TestStreamPreservationCommand',
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("🔍 Loading Test Modules...")
    
    # Import test modules
    try:
        from tests import (
            test_command_validation,
            test_stream_preservation_regression,
            test_enhanced_transcoding,
            test_stream_preservation
        )
        test_modules = [
            test_command_validation,
            test_stream_preservation_regression,
            test_enhanced_transcoding,
            test_stream_preservation
        ]
        print("✅ All test modules loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        return None
    
    print("\n📋 Building Test Suite by Priority...\n")
    
    # Add tests by priority
    def add_tests_by_pattern(patterns, priority_name):
        print(f"🔥 {priority_name} Tests:")
        added_count = 0
        
        for pattern in patterns:
            if '.' in pattern and len(pattern.split('.')) >= 3:
                # Specific test method
                module_name, class_name, method_name = pattern.split('.', 2)
                
                for module in test_modules:
                    if hasattr(module, class_name):
                        test_class = getattr(module, class_name)
                        if hasattr(test_class, method_name):
                            test_case = test_class(method_name)
                            suite.addTest(test_case)
                            print(f"   ✓ {class_name}.{method_name}")
                            added_count += 1
                            break
            else:
                # Test class or module
                for module in test_modules:
                    if hasattr(module, pattern):
                        test_class = getattr(module, pattern)
                        class_tests = loader.loadTestsFromTestCase(test_class)
                        suite.addTest(class_tests)
                        print(f"   ✓ {pattern} (all methods)")
                        added_count += class_tests.countTestCases()
                        break
        
        print(f"   📊 Added {added_count} tests\n")
        return added_count
    
    # Build suite by priority
    total_tests = 0
    total_tests += add_tests_by_pattern(critical_tests, "CRITICAL")
    total_tests += add_tests_by_pattern(high_priority_tests, "HIGH PRIORITY")
    total_tests += add_tests_by_pattern(medium_priority_tests, "MEDIUM PRIORITY")
    total_tests += add_tests_by_pattern(low_priority_tests, "LOW PRIORITY")
    
    print(f"🎯 Total Tests in Suite: {total_tests}")
    return suite


class VerboseTestResult(unittest.TextTestResult):
    """Custom test result class for detailed output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.critical_failures = []
        self.high_priority_failures = []
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        
        # Categorize failures by priority
        if any(critical in test_name for critical in [
            'test_command_contains_all_required_stream_preservation_flags',
            'test_vbr_function_uses_comprehensive_encoder',
            'test_end_to_end_command_generation_includes_streams'
        ]):
            self.critical_failures.append(test_name)
        elif any(high_priority in test_name for high_priority in [
            'test_command_structure_is_valid_ffmpeg_syntax',
            'test_no_conflicting_flags_present',
            'test_comprehensive_stream_mapping_pattern'
        ]):
            self.high_priority_failures.append(test_name)
    
    def addError(self, test, err):
        super().addError(test, err)
        # Same categorization logic for errors
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        
        if any(critical in test_name for critical in [
            'test_command_contains_all_required_stream_preservation_flags',
            'test_vbr_function_uses_comprehensive_encoder',
            'test_end_to_end_command_generation_includes_streams'
        ]):
            self.critical_failures.append(test_name)


class StreamPreservationTestRunner:
    """Specialized test runner for stream preservation tests."""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
    
    def run(self, suite):
        """Run the test suite with detailed reporting."""
        print("🚀 Starting Stream Preservation Test Suite")
        print("=" * 60)
        
        # Custom result class
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            resultclass=VerboseTestResult,
            stream=sys.stdout,
            buffer=True
        )
        
        # Run tests
        result = runner.run(suite)
        
        # Detailed reporting
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result):
        """Print detailed test summary."""
        print("\n" + "=" * 60)
        print("📊 STREAM PRESERVATION TEST SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total_tests - failures - errors
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failures}")
        print(f"💥 Errors: {errors}")
        print(f"📊 Success Rate: {(passed/total_tests)*100:.1f}%")
        
        # Critical failure analysis
        critical_failures = getattr(result, 'critical_failures', [])
        if critical_failures:
            print("\n🚨 CRITICAL FAILURES (Would NOT have caught original bug):")
            for failure in critical_failures:
                print(f"   💥 {failure}")
        else:
            print("\n✅ All CRITICAL tests passed (Original bug would be caught!)")
        
        # High priority failures
        high_priority_failures = getattr(result, 'high_priority_failures', [])
        if high_priority_failures:
            print("\n⚠️  HIGH PRIORITY FAILURES:")
            for failure in high_priority_failures:
                print(f"   ⚠️  {failure}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if failures + errors == 0:
            print("   🎉 All tests passing! Stream preservation is well protected.")
        else:
            if hasattr(result, 'critical_failures') and result.critical_failures:
                print("   🚨 Fix critical failures immediately - they indicate the original bug could still occur!")
            else:
                print("   ✅ Critical tests passing - original bug protection is in place.")
            
            if failures + errors > passed * 0.1:  # More than 10% failure rate
                print("   📋 Consider reviewing test implementations and mocking strategies.")
            else:
                print("   📈 Good test coverage with acceptable failure rate.")


def main():
    """Main entry point for running stream preservation tests."""
    print("🔧 Stream Preservation Bug Prevention Test Suite")
    print("=" * 60)
    print("This test suite focuses on preventing the stream preservation bug")
    print("that caused audio and subtitle streams to be missing from transcoded files.")
    print("=" * 60)
    
    # Create test suite
    suite = create_test_suite()
    if suite is None:
        print("❌ Failed to create test suite")
        return 1
    
    # Run tests
    runner = StreamPreservationTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    if result.failures or result.errors:
        critical_failures = getattr(result, 'critical_failures', [])
        if critical_failures:
            print("\n🚨 CRITICAL FAILURES DETECTED - Original bug protection may be incomplete!")
            return 2  # Critical failure
        else:
            print("\n⚠️  Some tests failed, but critical protection is in place.")
            return 1  # Non-critical failure
    else:
        print("\n🎉 All tests passed! Stream preservation is well protected.")
        return 0  # Success


if __name__ == '__main__':
    sys.exit(main())
