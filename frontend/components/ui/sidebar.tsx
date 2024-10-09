import Link from "next/link";

export function Sidebar() {
    return (
        <div className="bg-gray-800 text-white w-64 p-4">
            <h1 className="text-2xl font-bold mb-4">AI-EHR</h1>
            <nav>
                <ul>
                    <li className="mb-2">
                        <Link href="/" className="hover:text-gray-300">
                            Dashboard
                        </Link>
                    </li>
                    <li className="mb-2">
                        <Link href="/patients" className="hover:text-gray-300">
                            Patients
                        </Link>
                    </li>
                    <li className="mb-2">
                        <Link href="/schedule" className="hover:text-gray-300">
                            Schedule
                        </Link>
                    </li>
                </ul>
            </nav>
        </div>
    );
}