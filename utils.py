import winreg


def get_windows_default_font():
    try:
        reg_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
        i = 0
        fonts = []
        while True:
            try:
                font_name, font_file, _ = winreg.EnumValue(reg_key, i)
                fonts.append((font_name, font_file))
                i += 1
            except OSError:
                break
        winreg.CloseKey(reg_key)
        
        # Look for "Segoe UI" or "Segoe UI Variable"
        for font_name, font_file in fonts:
            if "Segoe UI" in font_name:
                return font_name
        return "Unknown"
    except Exception as e:
        return f"Error: {e}"


print("Windows Default Font:", get_windows_default_font())