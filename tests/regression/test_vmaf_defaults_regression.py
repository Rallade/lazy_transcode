"""
REGRESSION TEST: VMAF Default Values

Ensures that default VMAF targets remain at the user's preferred 95.0 value
and don't regress back to 92.0 or other values.

This test was created after a bug where defaults were accidentally changed
from 95.0 to 92.0 in multiple places.
"""

import unittest
import ast
import inspect
from pathlib import Path

from lazy_transcode.cli_enhanced import create_enhanced_parser
from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import optimize_encoder_settings_vbr_enhanced


class TestVMAFDefaultsRegression(unittest.TestCase):
    """
    CRITICAL REGRESSION TEST: Prevent VMAF default value changes.
    
    User prefers 95.0 as the default VMAF target, NOT 92.0.
    This test ensures defaults don't accidentally get changed.
    """
    
    def test_cli_enhanced_vmaf_default_is_95(self):
        """
        CLI ENHANCED DEFAULT: Ensure --vmaf-target defaults to 95.0
        
        The enhanced CLI should default to 95.0 for VMAF target.
        """
        parser = create_enhanced_parser()
        
        # Parse with no vmaf-target argument - should use default
        args = parser.parse_args([])
        
        self.assertEqual(args.vmaf_target, 95.0, 
                        "CLI Enhanced --vmaf-target default must be 95.0, not 92.0")
    
    def test_vbr_optimizer_enhanced_default_is_95(self):
        """
        VBR OPTIMIZER ENHANCED DEFAULT: Ensure function defaults to 95.0
        
        The enhanced VBR optimizer function should default to 95.0.
        """
        # Get the function signature
        sig = inspect.signature(optimize_encoder_settings_vbr_enhanced)
        target_vmaf_param = sig.parameters['target_vmaf']
        
        self.assertEqual(target_vmaf_param.default, 95.0,
                        "optimize_encoder_settings_vbr_enhanced target_vmaf default must be 95.0, not 92.0")
    
    def test_source_code_has_no_92_defaults(self):
        """
        SOURCE CODE SCAN: Ensure no user-facing defaults are set to 92.0
        
        Scan key source files to ensure 92.0 isn't used as a default where 95.0 should be.
        """
        files_to_check = [
            Path(__file__).parent.parent.parent / "lazy_transcode" / "cli_enhanced.py",
            Path(__file__).parent.parent.parent / "lazy_transcode" / "core" / "modules" / "optimization" / "vbr_optimizer_enhanced.py"
        ]
        
        problematic_patterns = []
        
        for file_path in files_to_check:
            if not file_path.exists():
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    # Check function definitions for 92.0 defaults
                    if isinstance(node, ast.FunctionDef):
                        for arg in node.args.defaults:
                            if isinstance(arg, ast.Constant) and arg.value == 92.0:
                                # Check if this is related to VMAF
                                func_source = ast.get_source_segment(content, node)
                                if func_source and 'vmaf' in func_source.lower():
                                    problematic_patterns.append(
                                        f"{file_path.name}:{node.lineno} - Function {node.name} has 92.0 default for VMAF parameter"
                                    )
                    
                    # Check argument parser calls
                    elif isinstance(node, ast.Call):
                        if (isinstance(node.func, ast.Attribute) and 
                            node.func.attr == 'add_argument' and
                            any(isinstance(kw.value, ast.Constant) and kw.value.value == 92.0 
                                for kw in node.keywords if kw.arg == 'default')):
                            # Check if this is vmaf-target
                            for arg in node.args:
                                if isinstance(arg, ast.Constant) and 'vmaf-target' in str(arg.value):
                                    problematic_patterns.append(
                                        f"{file_path.name}:{node.lineno} - CLI argument --vmaf-target has 92.0 default"
                                    )
                                    
            except SyntaxError:
                # Skip files with syntax errors
                continue
        
        self.assertEqual(len(problematic_patterns), 0,
                        f"Found user-facing VMAF defaults set to 92.0 instead of 95.0:\n" + 
                        "\n".join(problematic_patterns))
    
    def test_help_text_shows_95_not_92(self):
        """
        HELP TEXT CHECK: Ensure help text mentions 95.0 as default, not 92.0
        
        The CLI help should show the correct default value.
        """
        parser = create_enhanced_parser()
        help_text = parser.format_help()
        
        # Should mention 95.0 as default
        self.assertIn("default: 95.0", help_text,
                     "Help text should show 'default: 95.0' for VMAF target")
        
        # Should NOT mention 92.0 as default for VMAF
        if "default: 92.0" in help_text:
            # Check if it's specifically for VMAF target
            lines = help_text.split('\n')
            for i, line in enumerate(lines):
                if "default: 92.0" in line:
                    # Check surrounding context for VMAF
                    context_lines = lines[max(0, i-2):i+3]
                    context = ' '.join(context_lines).lower()
                    if 'vmaf' in context:
                        self.fail("Help text shows 'default: 92.0' for VMAF target - should be 95.0")


class TestVMAFStateManagementRegression(unittest.TestCase):
    """
    REGRESSION TEST: VMAF state management preserves user's target
    
    Ensures that resume functionality uses the original target VMAF,
    not a hardcoded default.
    """
    
    def test_state_manager_saves_target_vmaf(self):
        """
        STATE PERSISTENCE: Ensure VBROptimizationState saves target_vmaf
        
        When optimization is interrupted and resumed, it should use the
        original target VMAF, not a default.
        """
        from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import VBROptimizationState
        from pathlib import Path
        import tempfile
        import json
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "test_video.mkv"
            temp_file.touch()  # Create empty file
            
            # Create state manager with custom VMAF target
            state_manager = VBROptimizationState(
                temp_file, 
                target_vmaf=97.5,  # Custom value, not default
                vmaf_tolerance=0.8,
                encoder="hevc_nvenc",
                encoder_type="hardware"
            )
            
            # Save state
            state_manager.save_state()
            
            # Verify state file was created and contains correct values
            self.assertTrue(state_manager.state_file.exists())
            
            with open(state_manager.state_file, 'r') as f:
                saved_state = json.load(f)
            
            self.assertEqual(saved_state['target_vmaf'], 97.5,
                           "State should preserve original target_vmaf")
            self.assertEqual(saved_state['vmaf_tolerance'], 0.8,
                           "State should preserve original vmaf_tolerance")
            self.assertEqual(saved_state['encoder'], "hevc_nvenc",
                           "State should preserve original encoder")
            self.assertEqual(saved_state['encoder_type'], "hardware",
                           "State should preserve original encoder_type")


if __name__ == "__main__":
    unittest.main()
