import { useState, useEffect } from 'react';

interface ScreenSize {
  width: number;
  height: number;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLarge: boolean;
}

export const useResponsive = (): ScreenSize => {
  const [screenSize, setScreenSize] = useState<ScreenSize>({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768,
    isMobile: false,
    isTablet: false,
    isDesktop: false,
    isLarge: false,
  });

  useEffect(() => {
    const updateScreenSize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      setScreenSize({
        width,
        height,
        isMobile: width < 640,
        isTablet: width >= 640 && width < 1024,
        isDesktop: width >= 1024 && width < 1280,
        isLarge: width >= 1280,
      });
    };

    // Initial call
    updateScreenSize();

    // Add event listener
    window.addEventListener('resize', updateScreenSize);

    // Cleanup
    return () => window.removeEventListener('resize', updateScreenSize);
  }, []);

  return screenSize;
};

export const getResponsiveValue = <T,>(values: {
  mobile?: T;
  tablet?: T;
  desktop?: T;
  large?: T;
  default: T;
}, screenSize: ScreenSize): T => {
  if (screenSize.isLarge && values.large !== undefined) return values.large;
  if (screenSize.isDesktop && values.desktop !== undefined) return values.desktop;
  if (screenSize.isTablet && values.tablet !== undefined) return values.tablet;
  if (screenSize.isMobile && values.mobile !== undefined) return values.mobile;
  return values.default;
};

export const useResponsiveColumns = (baseColumns: number = 1): string => {
  const screenSize = useResponsive();
  
  return getResponsiveValue({
    mobile: `grid-cols-1`,
    tablet: `grid-cols-${Math.min(baseColumns + 1, 3)}`,
    desktop: `grid-cols-${Math.min(baseColumns + 2, 4)}`,
    large: `grid-cols-${Math.min(baseColumns + 3, 6)}`,
    default: `grid-cols-${baseColumns}`
  }, screenSize);
};

export const useResponsiveSpacing = (): {
  padding: string;
  margin: string;
  gap: string;
} => {
  const screenSize = useResponsive();
  
  return {
    padding: getResponsiveValue({
      mobile: 'p-2',
      tablet: 'p-4',
      desktop: 'p-6',
      large: 'p-8',
      default: 'p-4'
    }, screenSize),
    margin: getResponsiveValue({
      mobile: 'm-2',
      tablet: 'm-4',
      desktop: 'm-6',
      large: 'm-8',
      default: 'm-4'
    }, screenSize),
    gap: getResponsiveValue({
      mobile: 'gap-2',
      tablet: 'gap-4',
      desktop: 'gap-6',
      large: 'gap-8',
      default: 'gap-4'
    }, screenSize)
  };
};

export default useResponsive;
