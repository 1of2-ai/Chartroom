import ChartroomControl
import ChartroomMCP
import Foundation

@main
struct ChartroomMCPStdio {
    static func main() async {
        let session: ChartroomSession
        do {
            session = try ChartroomMCPFactory.defaultSession()
        } catch {
            fputs("chartroom-mcp: \(String(describing: error))\n", stderr)
            exit(1)
        }

        let server = ChartroomMCPServer(
            session: session,
            allowedLocalRoots: ChartroomMCPFactory.defaultAllowedLocalRoots()
        )

        while let line = readLine(strippingNewline: true) {
            do {
                if let response = try await server.handle(jsonLine: line) {
                    print(response)
                    fflush(stdout)
                }
            } catch {
                fputs("chartroom-mcp: \(String(describing: error))\n", stderr)
                fflush(stderr)
            }
        }
    }
}
