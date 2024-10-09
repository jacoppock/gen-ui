import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import React from 'react';

interface FHIRResourceDisplayProps {
    data: string;
}

const FHIRResourceDisplay: React.FC<FHIRResourceDisplayProps> = ({ data }) => {
    const parseData = (jsonString: string) => {
        try {
            return JSON.parse(jsonString.replace(/'/g, '"'));
        } catch (error) {
            console.error("Error parsing JSON:", error);
            return null;
        }
    };

    const renderPatient = (patient: any) => (
        <>
            <CardTitle>{`${patient.name[0].given.join(' ')} ${patient.name[0].family}`}</CardTitle>
            <CardDescription>
                <p>Gender: {patient.gender}</p>
                <p>Birth Date: {patient.birthDate}</p>
                <p>ID: {patient.id}</p>
            </CardDescription>
        </>
    );

    const renderGenericResource = (resource: any) => (
        <>
            <CardTitle>{resource.resourceType}</CardTitle>
            <CardDescription>
                {Object.entries(resource).map(([key, value]) => (
                    <p key={key}>{`${key}: ${JSON.stringify(value)}`}</p>
                ))}
            </CardDescription>
        </>
    );

    const parsedData = parseData(data);

    if (!parsedData) {
        return <div>Error parsing data</div>;
    }

    const renderContent = () => {
        switch (parsedData.resourceType) {
            case 'Patient':
                return renderPatient(parsedData);
            default:
                return renderGenericResource(parsedData);
        }
    };

    return (
        <Card className="w-[450px]">
            <CardHeader className="items-start space-y-1">
                {renderContent()}
            </CardHeader>
            <CardContent>
                <div className="text-sm text-muted-foreground">
                    Resource Type: {parsedData.resourceType}
                </div>
            </CardContent>
        </Card>
    );
};

export default FHIRResourceDisplay;