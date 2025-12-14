"""
Example Extension for MedICS

This is a template extension that demonstrates how to create
independent MedICS extensions as pip packages.
"""

from medics_extension_sdk import BaseExtension
from .ui.main_widget import ExampleExtensionWidget
import logging
__version__ = "1.0.0"


class ExampleExtension(BaseExtension):
    """Example extension demonstrating the MedICS plugin system.
    
    This extension shows how to:
    - Inherit from BaseExtension
    - Implement required methods
    - Create UI widgets
    - Access application context
    - Use configuration
    - Log messages
    """
    
    def __init__(self):
        """Initialize the example extension."""
        super().__init__(
            extension_name="Example Extension",
            author_name="Your Name"
        )
        self.widget = None
        self.logger = logging.getLogger(__name__)
    def get_version(self) -> str:
        """Return the extension version.
        
        Returns:
            str: Version string (e.g., "1.0.0")
        """
        return __version__
    
    def get_description(self) -> str:
        """Return a short description of the extension.
        
        Returns:
            str: Brief description shown in extension manager
        """
        return "A template extension demonstrating MedICS plugin system"
    
    def get_category(self) -> str:
        """Return the extension category for UI grouping.
        
        Returns:
            str: Category name (e.g., "Image Analysis", "Data Processing")
        """
        return "Examples"
    
    def initialize(self, app_context) -> bool:
        """Initialize the extension with application context.
        
        This is called once when the extension is loaded. Use it to:
        - Store the app_context
        - Access other components
        - Load configuration
        - Set up resources
        
        Args:
            app_context: Main application context providing access to:
                - workspace_manager: Data and workspace management
                - config_manager: Configuration access
                - theme_manager: UI theming
                - extension_manager: Extension management
                - main_window: Main application window
                - root_path: Application root directory
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Store app context
        self.app_context = app_context
        
        # Access components
        self.workspace_manager = app_context.get_component("workspace_manager")
        self.config_manager = app_context.get_component("config_manager")
        
        # Load configuration
        self.load_configuration()
        
        # Log initialization
        self.logger.info(f"Example Extension v{self.get_version()} initialized")
        
        return True
    
    def load_configuration(self):
        """Load extension-specific configuration."""
        if self.config_manager:
            # Read settings with defaults
            self.setting1 = self.config_manager.get_value(
                "ExampleExtension", "setting1", "default_value"
            )
            self.setting2 = self.config_manager.get_value(
                "ExampleExtension", "setting2", "default_value"
            )
            
            self.logger.debug(f"Configuration loaded: setting1={self.setting1}")
    
    def save_configuration(self):
        """Save extension-specific configuration."""
        if self.config_manager:
            self.config_manager.set_value(
                "ExampleExtension", "setting1", self.setting1
            )
            self.config_manager.set_value(
                "ExampleExtension", "setting2", self.setting2
            )
            # Save only if the method exists (some versions may not have it)
            if hasattr(self.config_manager, 'save'):
                self.config_manager.save()
            
            self.logger.debug("Configuration saved")
    
    def create_widget(self, parent=None, **kwargs):
        """Create the main extension widget.
        
        This is called when the user opens the extension from the menu.
        
        Args:
            parent: Parent Qt widget (usually None for top-level)
            **kwargs: Additional parameters passed to the widget
        
        Returns:
            QWidget: The main extension widget
        """
        self.widget = ExampleExtensionWidget(parent, self, **kwargs)
        return self.widget
    
    def cleanup(self) -> None:
        """Clean up resources before unload.
        
        This is called when:
        - The extension is unloaded
        - The application is closing
        
        Use it to:
        - Close files
        - Stop threads
        - Release resources
        - Save state
        """
        # Save configuration
        self.save_configuration()
        
        # Clean up widget
        if self.widget:
            self.widget.cleanup()
            self.widget = None
        
        # Log cleanup
        self.logger.info("Example Extension cleaned up")


# Export extension class for entry point
__all__ = ['ExampleExtension']
