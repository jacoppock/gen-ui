import React from 'react'

interface ResponsePanelProps {
    dashboard: React.ReactElement | null
}

export function ResponsePanel({ dashboard }: ResponsePanelProps) {
    return (
        <div className="h-full flex flex-col">
            <h2 className="text-2xl font-bold mb-4 text-gray-800">Patient Profile</h2>
            {dashboard ? (
                <div className="flex-grow overflow-y-auto bg-white rounded-lg shadow-inner p-4">
                    {dashboard}
                </div>
            ) : (
                <div className="flex-grow flex items-center justify-center bg-gray-100 rounded-lg">
                    <p className="text-gray-500 text-center">
                        No patient profile generated yet. <br />
                        Please start the onboarding process in the chat window.
                    </p>
                </div>
            )}
        </div>
    )
}