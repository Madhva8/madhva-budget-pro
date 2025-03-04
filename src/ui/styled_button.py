#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Styled Button Module

This module defines a customized button with better visual styling
than the standard QPushButton.
"""

from PyQt5.QtWidgets import (
    QPushButton, QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QPoint, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPalette, QFont


class StyledButton(QPushButton):
    """Custom styled button with macOS-inspired appearance."""

    def __init__(self, text="", icon=None, is_secondary=False, parent=None):
        """
        Initialize the styled button.

        Args:
            text: Button text
            icon: Button icon (QIcon)
            is_secondary: Whether this is a secondary button
            parent: Parent widget
        """
        super().__init__(text, parent)

        # Store properties
        self.is_secondary = is_secondary
        self.hover_state = False
        self.click_animation = None
        
        # Use default system font
        font = self.font()
        font.setWeight(QFont.Medium)
        self.setFont(font)

        # Set icon if provided
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(16, 16))
        
        # No shadow effect for better compatibility
        
        # Apply styling
        self.apply_styles()
        
        # Connect hover events
        self.setMouseTracking(True)
        self.enterEvent = self.on_enter
        self.leaveEvent = self.on_leave

    def setupAnimation(self):
        """Setup click animation for the button."""
        self.click_animation = QPropertyAnimation(self, b"pos")
        self.click_animation.setDuration(100)
        self.click_animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def mousePressEvent(self, event):
        """Handle mouse press with simpler effect."""
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        super().mouseReleaseEvent(event)
        
    def on_enter(self, event):
        """Handle mouse enter event."""
        self.hover_state = True
        self.update()
        
    def on_leave(self, event):
        """Handle mouse leave event."""
        self.hover_state = False
        self.update()

    def apply_styles(self):
        """Apply custom styles to the button."""
        # Set default styles based on macOS design
        if self.is_secondary:
            # Secondary button (outline style)
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.8);
                    color: #007AFF;
                    border: 1px solid #DDDDDD;
                    border-radius: 6px;
                    padding: 8px 16px;
                }

                QPushButton:hover {
                    background-color: rgba(0, 122, 255, 0.05);
                    border: 1px solid #BBBBBB;
                }

                QPushButton:pressed {
                    background-color: rgba(0, 122, 255, 0.1);
                }

                QPushButton:disabled {
                    border: 1px solid #E5E5E5;
                    color: #AAAAAA;
                    background-color: #F5F5F5;
                }
            """)
        else:
            # Primary button (filled style)
            self.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                }

                QPushButton:hover {
                    background-color: #0066DD;
                }

                QPushButton:pressed {
                    background-color: #0055CC;
                }

                QPushButton:disabled {
                    background-color: #B0D0FF;
                }
            """)

    def setSecondary(self, is_secondary):
        """
        Set the button as primary or secondary.

        Args:
            is_secondary: Whether this is a secondary button
        """
        self.is_secondary = is_secondary
        self.apply_styles()

    def paintEvent(self, event):
        """
        Custom paint event for the button.

        Args:
            event: Paint event
        """
        # Call the parent class paint event for basic button drawing
        super().paintEvent(event)


class DangerButton(StyledButton):
    """Styled button for dangerous actions like deletion."""

    def __init__(self, text="", icon=None, parent=None):
        """
        Initialize the danger button.

        Args:
            text: Button text
            icon: Button icon (QIcon)
            parent: Parent widget
        """
        super().__init__(text, icon, False, parent)

        # Apply danger styling
        self.apply_danger_styles()

    def apply_danger_styles(self):
        """Apply danger-specific styles."""
        self.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #E0352B;
            }

            QPushButton:pressed {
                background-color: #C02F26;
            }

            QPushButton:disabled {
                background-color: #FFBFBC;
            }
        """)


class SuccessButton(StyledButton):
    """Styled button for positive actions like confirmation."""

    def __init__(self, text="", icon=None, parent=None):
        """
        Initialize the success button.

        Args:
            text: Button text
            icon: Button icon (QIcon)
            parent: Parent widget
        """
        super().__init__(text, icon, False, parent)

        # Apply success styling
        self.apply_success_styles()

    def apply_success_styles(self):
        """Apply success-specific styles."""
        self.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #30B350;
            }

            QPushButton:pressed {
                background-color: #2A9F47;
            }

            QPushButton:disabled {
                background-color: #B8ECC5;
            }
        """)


class IconButton(QPushButton):
    """Icon-only button with a macOS-inspired look."""

    def __init__(self, icon, tooltip="", parent=None):
        """
        Initialize the icon button.

        Args:
            icon: Button icon (QIcon)
            tooltip: Button tooltip
            parent: Parent widget
        """
        super().__init__(parent)

        # Set icon
        self.setIcon(icon)
        self.setIconSize(QSize(20, 20))

        # Set tooltip
        if tooltip:
            self.setToolTip(tooltip)

        # Set fixed size
        self.setFixedSize(QSize(36, 36))

        # Setup shadow effect
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(6)
        self.shadow.setColor(QColor(0, 0, 0, 25))
        self.shadow.setOffset(0, 1)
        self.setGraphicsEffect(self.shadow)

        # Track hover state
        self.hover = False
        self.setMouseTracking(True)
        
        # Apply styling
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.7);
                border: 1px solid #E5E5E5;
                border-radius: 18px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid #D0D0D0;
            }

            QPushButton:pressed {
                background-color: rgba(240, 240, 240, 1.0);
                border: 1px solid #C0C0C0;
            }
        """)
        
    def enterEvent(self, event):
        """Handle mouse enter event."""
        self.hover = True
        self.shadow.setBlurRadius(8)
        self.shadow.setOffset(0, 2)
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave event."""
        self.hover = False
        self.shadow.setBlurRadius(6)
        self.shadow.setOffset(0, 1)
        self.update()
        super().leaveEvent(event)