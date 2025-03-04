#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Styled Button Module

This module defines customized buttons with macOS-style appearances using PySide6.
"""

from PySide6.QtWidgets import (
    QPushButton, QGraphicsDropShadowEffect, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QPoint, QTimer, Property, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QFont


class ModernStyledButton(QPushButton):
    """Custom styled button with authentic macOS appearance."""

    def __init__(self, text="", icon=None, is_secondary=False, parent=None):
        """
        Initialize the styled button.

        Args:
            text: Button text
            icon: Button icon
            is_secondary: Whether this is a secondary button
            parent: Parent widget
        """
        super().__init__(text, parent)

        # Store properties
        self.is_secondary = is_secondary
        self.hover_state = False
        
        # Set size policy for proper macOS button sizing and better adaptability
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumHeight(22)  # macOS standard button height
        self.setMinimumWidth(80)   # Minimum width to prevent buttons from disappearing
        
        # Set icon if provided
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(16, 16))

        # Apply macOS-style button settings
        self.setCursor(Qt.PointingHandCursor)
        self.apply_styles()
        
        # Setup hover detection
        self.setMouseTracking(True)

    def apply_styles(self):
        """Apply custom styles to match macOS buttons."""
        if self.is_secondary:
            # Secondary button (outline style)
            self.setStyleSheet("""
                QPushButton {
                    background-color: var(--button-secondary-bg, rgba(255, 255, 255, 0.5));
                    color: var(--button-secondary-text, #007AFF);
                    border: 0.5px solid var(--button-secondary-border, #DDDDDD);
                    border-radius: 4px;
                    padding: 3px 10px;
                    outline: none;
                }

                QPushButton:hover {
                    background-color: var(--button-secondary-hover-bg, rgba(0, 122, 255, 0.05));
                    border: 0.5px solid var(--button-secondary-hover-border, #BBBBBB);
                }

                QPushButton:pressed {
                    background-color: var(--button-secondary-pressed-bg, rgba(0, 122, 255, 0.1));
                }

                QPushButton:disabled {
                    color: var(--button-disabled-text, #AAAAAA);
                    background-color: var(--button-disabled-bg, rgba(240, 240, 240, 0.8));
                    border: 0.5px solid var(--button-disabled-border, #E5E5E5);
                }
            """)
            
            # Apply icon color based on theme
            if self.icon():
                theme = self.palette().color(QPalette.Window).lightness() < 128
                if theme:  # Dark theme
                    color = QColor(255, 255, 255)  # White icons for dark theme
                else:
                    color = QColor(0, 0, 0)  # Black icons for light theme
                
                # Update icon with appropriate color
                icon = self.icon()
                # If there's a way to get the icon's pixmap and adjust color, it would go here
                # But for now, we're relying on the theme variables to handle icon colors
        else:
            # Primary button (accent colored)
            self.setStyleSheet("""
                QPushButton {
                    background-color: var(--button-primary-bg, #007AFF);
                    color: var(--button-primary-text, white);
                    border: none;
                    border-radius: 4px;
                    padding: 3px 10px;
                    outline: none;
                }

                QPushButton:hover {
                    background-color: var(--button-primary-hover-bg, #0062CC);
                }

                QPushButton:pressed {
                    background-color: var(--button-primary-pressed-bg, #0051A8);
                }

                QPushButton:disabled {
                    background-color: var(--button-disabled-accent-bg, #B0D0FF);
                }
            """)
            
            # Icon color is white for primary buttons regardless of theme
            # because primary buttons have a colored background

    def setSecondary(self, is_secondary):
        """
        Set the button as primary or secondary.

        Args:
            is_secondary: Whether this is a secondary button
        """
        self.is_secondary = is_secondary
        self.apply_styles()

    def enterEvent(self, event):
        """Handle mouse enter event with subtle scale effect."""
        self.hover_state = True
        # Create subtle grow effect
        animation = QPropertyAnimation(self, b"maximumSize")
        animation.setDuration(100)
        animation.setStartValue(self.size())
        animation.setEndValue(QSize(self.width() + 1, self.height() + 1))
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start(QPropertyAnimation.DeleteWhenStopped)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave event by restoring original size."""
        self.hover_state = False
        # Restore size
        animation = QPropertyAnimation(self, b"maximumSize")
        animation.setDuration(100)
        animation.setStartValue(self.size())
        animation.setEndValue(QSize(self.maximumWidth() - 1, self.maximumHeight() - 1))
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start(QPropertyAnimation.DeleteWhenStopped)
        super().leaveEvent(event)


class ModernDangerButton(ModernStyledButton):
    """Styled button for dangerous actions like deletion with macOS red accent."""

    def __init__(self, text="", icon=None, parent=None):
        """
        Initialize the danger button.

        Args:
            text: Button text
            icon: Button icon
            parent: Parent widget
        """
        super().__init__(text, icon, False, parent)

        # Apply danger styling
        self.apply_danger_styles()

    def apply_danger_styles(self):
        """Apply danger-specific styles using macOS system red."""
        self.setStyleSheet("""
            QPushButton {
                background-color: var(--button-danger-bg, #FF3B30);
                color: var(--button-danger-text, white);
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                outline: none;
                font-weight: 500;
            }

            QPushButton:hover {
                background-color: var(--button-danger-hover-bg, #E0342B);
            }

            QPushButton:pressed {
                background-color: var(--button-danger-pressed-bg, #C12E26);
            }

            QPushButton:disabled {
                background-color: var(--button-danger-disabled-bg, #FFBFBC);
                color: rgba(255, 255, 255, 0.6);
            }
        """)
        
    def setEnabled(self, enabled):
        """Override setEnabled to make disabled state more visible."""
        super().setEnabled(enabled)
        # Make sure the color change is applied immediately by refreshing style
        if enabled:
            self.setStyleSheet(self.styleSheet())
            # Apply a subtle pop effect when enabled
            animation = QPropertyAnimation(self, b"maximumSize")
            animation.setDuration(100)
            animation.setStartValue(self.size())
            animation.setEndValue(QSize(self.width() + 2, self.height() + 2))
            animation.setEasingCurve(QEasingCurve.OutCubic)
            animation.start(QPropertyAnimation.DeleteWhenStopped)
            
            # Return to normal size
            QTimer.singleShot(150, lambda: self.setMaximumSize(self.width(), self.height()))


class ModernSuccessButton(ModernStyledButton):
    """Styled button for positive actions like confirmation with macOS green accent."""

    def __init__(self, text="", icon=None, parent=None):
        """
        Initialize the success button.

        Args:
            text: Button text
            icon: Button icon
            parent: Parent widget
        """
        super().__init__(text, icon, False, parent)

        # Apply success styling
        self.apply_success_styles()

    def apply_success_styles(self):
        """Apply success-specific styles using macOS system green."""
        self.setStyleSheet("""
            QPushButton {
                background-color: var(--button-success-bg, #34C759);
                color: var(--button-success-text, white);
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                outline: none;
            }

            QPushButton:hover {
                background-color: var(--button-success-hover-bg, #30B350);
            }

            QPushButton:pressed {
                background-color: var(--button-success-pressed-bg, #2A9F47);
            }

            QPushButton:disabled {
                background-color: var(--button-success-disabled-bg, #B8ECC5);
            }
        """)


class ModernIconButton(QPushButton):
    """Icon-only button with macOS-style appearance for toolbars and actions."""

    def __init__(self, icon, tooltip="", parent=None):
        """
        Initialize the icon button.

        Args:
            icon: Button icon
            tooltip: Button tooltip
            parent: Parent widget
        """
        super().__init__(parent)

        # Set icon
        self.setIcon(icon)
        self.setIconSize(QSize(18, 18))

        # Set tooltip
        if tooltip:
            self.setToolTip(tooltip)

        # Set fixed size for consistent appearance
        self.setFixedSize(QSize(28, 28))
        
        # Set cursor
        self.setCursor(Qt.PointingHandCursor)
        
        # Apply styling
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 14px;
                padding: 5px;
                color: var(--text-color, black); /* This affects icon color in most themes */
            }

            QPushButton:hover {
                background-color: var(--button-hover-bg, rgba(0, 0, 0, 0.05));
            }

            QPushButton:pressed {
                background-color: var(--button-pressed-bg, rgba(0, 0, 0, 0.1));
            }
        """)
    
    def changeEvent(self, event):
        """Update icon color when theme or palette changes."""
        super().changeEvent(event)
        # Check if the event was a palette change 
        if event.type() == event.PaletteChange:
            # Get the theme by checking if the window background is dark
            is_dark = self.palette().color(QPalette.Window).lightness() < 128
            # We can't easily adjust the icon color here, but the stylesheet
            # should be using CSS variables that will update automatically