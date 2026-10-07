import AppKit
import Foundation

let app = NSApplication.shared
app.setActivationPolicy(.regular)
app.activate(ignoringOtherApps: true)
let panel = NSOpenPanel()
panel.title = "เลือกโฟลเดอร์รูป — Face Sorter"
panel.prompt = "เลือกโฟลเดอร์นี้"
panel.canChooseDirectories = true
panel.canChooseFiles = false
panel.allowsMultipleSelection = false
panel.canCreateDirectories = false
if CommandLine.arguments.count > 1 {
    panel.directoryURL = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
}
let response = panel.runModal()
let folder: Any = response == .OK ? (panel.url?.path as Any? ?? NSNull()) : NSNull()
let data = try JSONSerialization.data(withJSONObject: ["folder": folder])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
