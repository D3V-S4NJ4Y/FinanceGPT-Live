import React from 'react';
import { useResponsive, useResponsiveSpacing } from '../../hooks/useResponsive';

interface ResponsiveContainerProps {
  children: React.ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
  padding?: boolean;
}

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({
  children,
  className = '',
  maxWidth = 'xl',
  padding = true
}) => {
  const { padding: responsivePadding } = useResponsiveSpacing();
  
  const maxWidthClass = {
    sm: 'max-w-sm',
    md: 'max-w-md', 
    lg: 'max-w-4xl',
    xl: 'max-w-7xl',
    '2xl': 'max-w-full',
    full: 'w-full'
  }[maxWidth];

  return (
    <div className={`
      ${maxWidthClass} mx-auto 
      ${padding ? responsivePadding : ''} 
      ${className}
    `}>
      {children}
    </div>
  );
};

interface ResponsiveGridProps {
  children: React.ReactNode;
  columns?: {
    mobile?: number;
    tablet?: number;
    desktop?: number;
    large?: number;
  };
  gap?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  children,
  columns = { mobile: 1, tablet: 2, desktop: 3, large: 4 },
  gap = 'md',
  className = ''
}) => {
  const screenSize = useResponsive();
  
  const getColumnsClass = () => {
    if (screenSize.isLarge && columns.large) return `grid-cols-${columns.large}`;
    if (screenSize.isDesktop && columns.desktop) return `grid-cols-${columns.desktop}`;
    if (screenSize.isTablet && columns.tablet) return `grid-cols-${columns.tablet}`;
    if (screenSize.isMobile && columns.mobile) return `grid-cols-${columns.mobile}`;
    return 'grid-cols-1';
  };
  
  const gapClass = {
    sm: 'gap-2 sm:gap-3',
    md: 'gap-3 sm:gap-4 lg:gap-6',
    lg: 'gap-4 sm:gap-6 lg:gap-8'
  }[gap];

  return (
    <div className={`grid ${getColumnsClass()} ${gapClass} ${className}`}>
      {children}
    </div>
  );
};

interface ResponsiveCardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
  glass?: boolean;
}

export const ResponsiveCard: React.FC<ResponsiveCardProps> = ({
  children,
  className = '',
  padding = 'md',
  glass = true
}) => {
  const paddingClass = {
    sm: 'p-3 sm:p-4',
    md: 'p-4 sm:p-6',
    lg: 'p-6 sm:p-8'
  }[padding];
  
  const baseClasses = glass 
    ? 'bg-black/40 backdrop-blur-sm border border-gray-700'
    : 'bg-gray-800 border border-gray-700';

  return (
    <div className={`
      ${baseClasses} 
      rounded-lg sm:rounded-xl 
      ${paddingClass} 
      transition-all duration-200 
      hover:border-gray-600
      ${className}
    `}>
      {children}
    </div>
  );
};

interface ResponsiveFlexProps {
  children: React.ReactNode;
  direction?: 'row' | 'col';
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  gap?: 'sm' | 'md' | 'lg';
  wrap?: boolean;
  className?: string;
}

export const ResponsiveFlex: React.FC<ResponsiveFlexProps> = ({
  children,
  direction = 'col',
  align = 'start',
  justify = 'start',
  gap = 'md',
  wrap = false,
  className = ''
}) => {
  const directionClass = direction === 'row' 
    ? 'flex-col sm:flex-row' 
    : 'flex-col';
    
  const alignClass = `items-${align}`;
  const justifyClass = `justify-${justify}`;
  
  const gapClass = {
    sm: 'gap-2 sm:gap-3',
    md: 'gap-3 sm:gap-4',
    lg: 'gap-4 sm:gap-6'
  }[gap];
  
  const wrapClass = wrap ? 'flex-wrap' : '';

  return (
    <div className={`
      flex ${directionClass} ${alignClass} ${justifyClass} 
      ${gapClass} ${wrapClass} ${className}
    `}>
      {children}
    </div>
  );
};

export default ResponsiveContainer;
